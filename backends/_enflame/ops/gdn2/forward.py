"""Chunked GDN2 forward implemented with Torch GCU operators.

The implementation is an isolated, inference-only experiment. K1 prepares
all chunks in parallel with batched matrix operations. K2 keeps only the
inter-chunk state recurrence serial, matching the structure of the S60 fused
GDN2 path without invoking Triton or TopsCC kernels directly.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

try:
    import torch_gcu as _torch_gcu
except ImportError:
    _torch_gcu = None

from .validation import validate_gdn2_torch_gcu_inputs


@lru_cache(maxsize=8)
def _chunk_constants(
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache read-only chunk constants for repeated inference calls."""
    identity = torch.eye(
        chunk_size,
        dtype=torch.float16,
        device=device,
    )
    positions = torch.arange(chunk_size, device=device, dtype=torch.int32)
    rows = positions.reshape(1, chunk_size, 1)
    cols = positions.reshape(1, 1, chunk_size)
    return identity, rows >= cols, rows > cols


def _initial_state_kv(
    initial_state: torch.Tensor | None,
    *,
    B: int,
    H: int,
    K: int,
    V: int,
    device: torch.device,
    state_v_first: bool,
) -> torch.Tensor:
    if initial_state is None:
        return torch.zeros(B * H, K, V, dtype=torch.float32, device=device)
    state = initial_state.transpose(-1, -2) if state_v_first else initial_state
    return state.contiguous().reshape(B * H, K, V)


def _pad_h_major_chunks(
    tensor: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Convert [B, T, H, D] to contiguous [B*H, NT, BT, D]."""
    B, T, H, D = tensor.shape
    NT = (T + chunk_size - 1) // chunk_size
    padded_t = NT * chunk_size
    tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(B * H, T, D)
    if padded_t != T:
        tensor = F.pad(tensor, (0, 0, 0, padded_t - T))
    return tensor.reshape(B * H, NT, chunk_size, D)


def _triangular_inverse(lower: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Compute (I + L)^-1 for a strictly-lower nilpotent matrix L."""
    matrix_count = lower.shape[0]
    lower_power = lower.to(torch.float16)
    identity, _causal_mask, _strict_causal_mask = _chunk_constants(
        lower.device,
        chunk_size,
    )
    identity = identity.expand(matrix_count, -1, -1)
    inverse = identity - lower_power

    power = 2
    while power < chunk_size:
        lower_power = torch.bmm(lower_power, lower_power)
        inverse = inverse + torch.bmm(inverse, lower_power)
        power *= 2
    return inverse.to(lower.dtype)


def _activate_gate(
    g: torch.Tensor,
    *,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    use_gate_in_kernel: bool,
) -> torch.Tensor:
    if not use_gate_in_kernel:
        return g.float()

    H, K = g.shape[-2:]
    gate = g.float()
    if dt_bias is not None:
        gate = gate + dt_bias.reshape(1, 1, H, K)
    rate = torch.exp(A_log).reshape(1, 1, H, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(rate * gate)
    return -rate * F.softplus(gate)


def _build_pair_matrices(
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    b_chunks: torch.Tensor,
    g_cumsum: torch.Tensor,
    qg: torch.Tensor,
    erase_key: torch.Tensor,
    k_from_start: torch.Tensor | None,
    *,
    scale: float,
    stable_scores: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build causal QK and erase-KK matrices without unsafe exponentials."""
    BH, NT, BT, K = q_chunks.shape
    matrix_count = BH * NT
    q_flat = q_chunks.reshape(matrix_count, BT, K)
    k_flat = k_chunks.reshape(matrix_count, BT, K)
    b_flat = b_chunks.reshape(matrix_count, BT, K)
    g_flat = g_cumsum.reshape(matrix_count, BT, K)
    qg_flat = qg.reshape(matrix_count, BT, K)
    erase_key_flat = erase_key.reshape(matrix_count, BT, K)

    if stable_scores:
        # A safe gate can contribute as much as -5 per token. A single
        # exp(-g_cumsum) over BT=64 would overflow FP32. Factor each retained
        # lower-triangular block around its row-block boundary so both
        # exponentials are non-positive. Sixteen rows keep the local span in
        # range even at the supported lower bound.
        block_size = 16
        Aqk = torch.zeros(
            matrix_count,
            BT,
            BT,
            dtype=torch.float32,
            device=q_chunks.device,
        )
        Akk = torch.zeros_like(Aqk)
        for row_start in range(0, BT, block_size):
            row_end = min(row_start + block_size, BT)
            shift = g_flat[:, row_start : row_start + 1]
            row_gate = torch.exp(g_flat[:, row_start:row_end] - shift)
            q_row = (q_flat[:, row_start:row_end].float() * row_gate).to(dtype)
            erase_row = (
                k_flat[:, row_start:row_end].float()
                * b_flat[:, row_start:row_end].float()
                * row_gate
            ).to(dtype)
            for col_start in range(0, row_start + 1, block_size):
                col_end = min(col_start + block_size, BT)
                col_gate = torch.exp(shift - g_flat[:, col_start:col_end])
                k_col = (
                    k_flat[:, col_start:col_end].float() * col_gate
                ).to(dtype)
                Aqk[:, row_start:row_end, col_start:col_end] = torch.bmm(
                    q_row,
                    k_col.transpose(1, 2),
                ).float()
                Akk[:, row_start:row_end, col_start:col_end] = torch.bmm(
                    erase_row,
                    k_col.transpose(1, 2),
                ).float()
    else:
        if k_from_start is None:
            raise RuntimeError("default pair path requires k_from_start")
        key_t = k_from_start.reshape(matrix_count, BT, K).transpose(1, 2)
        Aqk = torch.bmm(qg_flat, key_t).float()
        Akk = torch.bmm(erase_key_flat, key_t).float()

    _identity, causal_mask, strict_causal_mask = _chunk_constants(
        q_chunks.device,
        BT,
    )
    Aqk = torch.where(causal_mask, Aqk * scale, 0.0).to(dtype)
    Akk = torch.where(strict_causal_mask, Akk, 0.0).to(dtype)
    return Aqk, Akk


def _build_chunk_workspaces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    stable_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run parallel K1 and return chunk-major workspaces consumed by K2."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    BT = chunk_size
    NT = (T + BT - 1) // BT
    matrix_count = BH * NT
    # q/k may be FP32 after optional L2 normalization. Keep all matrix-engine
    # operands in the public low-precision input dtype, matching the fused path.
    dtype = v.dtype

    q_chunks = _pad_h_major_chunks(q, chunk_size=BT)
    k_chunks = _pad_h_major_chunks(k, chunk_size=BT)
    v_chunks = _pad_h_major_chunks(v, chunk_size=BT)
    g_chunks = _pad_h_major_chunks(g, chunk_size=BT)
    b_chunks = _pad_h_major_chunks(b, chunk_size=BT)
    w_chunks = _pad_h_major_chunks(w, chunk_size=BT)

    g_cumsum = torch.cumsum(g_chunks.float(), dim=2)
    g_last = g_cumsum[:, :, -1]
    gate_from_start = torch.exp(g_cumsum)

    qg = (q_chunks.float() * gate_from_start).to(dtype)
    erase_key = (
        k_chunks.float() * b_chunks.float() * gate_from_start
    ).to(dtype)
    k_from_start_float = None
    k_from_start = None
    if not stable_scores:
        k_from_start_float = k_chunks.float() * torch.exp(-g_cumsum)
        k_from_start = k_from_start_float.to(dtype)

    erase_key_flat = erase_key.reshape(matrix_count, BT, K)
    Aqk, Akk = _build_pair_matrices(
        q_chunks,
        k_chunks,
        b_chunks,
        g_cumsum,
        qg,
        erase_key,
        k_from_start,
        scale=scale,
        stable_scores=stable_scores,
        dtype=dtype,
    )
    state_decay_base = torch.exp(g_last)
    if stable_scores:
        kg = (
            k_chunks.float() * torch.exp(g_last.unsqueeze(2) - g_cumsum)
        ).to(dtype)
    else:
        if k_from_start_float is None:
            raise RuntimeError("default kg path requires k_from_start_float")
        kg = (
            k_from_start_float * state_decay_base.unsqueeze(2)
        ).to(dtype)
    del k_from_start_float, k_from_start

    Akk_inverse = _triangular_inverse(Akk, chunk_size=BT)

    write_value = (v_chunks.float() * w_chunks.float()).to(dtype)
    u_wy = torch.bmm(
        Akk_inverse,
        write_value.reshape(matrix_count, BT, V),
    ).reshape(BH, NT, BT, V)
    w_wy = torch.bmm(
        Akk_inverse,
        erase_key_flat,
    ).reshape(BH, NT, BT, K)

    wq = torch.cat((w_wy, qg), dim=2).permute(1, 0, 2, 3).contiguous()
    out_state_a = (
        torch.cat(
            (
                Aqk.reshape(BH, NT, BT, BT),
                kg.transpose(-1, -2),
            ),
            dim=2,
        )
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    u_wy = u_wy.permute(1, 0, 2, 3).contiguous()
    state_decay = state_decay_base.permute(1, 0, 2).contiguous()
    return wq, out_state_a, u_wy, state_decay


def _run_chunked_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    chunk_size: int,
    stable_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run equal-length sequences with one BH-batched chunk recurrence."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    BT = chunk_size
    NT = (T + BT - 1) // BT

    wq, out_state_a, u_wy, state_decay = _build_chunk_workspaces(
        q,
        k,
        v,
        g,
        b,
        w,
        scale=scale,
        chunk_size=BT,
        stable_scores=stable_scores,
    )
    state = _initial_state_kv(
        initial_state,
        B=B,
        H=H,
        K=K,
        V=V,
        device=q.device,
        state_v_first=state_v_first,
    )
    output_chunks = torch.empty(
        NT,
        BH,
        BT,
        V,
        dtype=v.dtype,
        device=q.device,
    )

    for chunk_id in range(NT):
        state_projection = torch.bmm(
            wq[chunk_id],
            state.to(v.dtype),
        ).float()
        kh = state_projection[:, :BT]
        qh = state_projection[:, BT:]
        v_new = (u_wy[chunk_id].float() - kh).to(v.dtype)

        output_state_update = torch.bmm(
            out_state_a[chunk_id],
            v_new,
        ).float()
        output_correction = output_state_update[:, :BT]
        state_update = output_state_update[:, BT:]
        output_chunks[chunk_id] = torch.add(
            output_correction,
            qh,
            alpha=scale,
        ).to(v.dtype)
        state = torch.addcmul(
            state_update,
            state,
            state_decay[chunk_id].unsqueeze(-1),
        )

    output = output_chunks.reshape(NT, B, H, BT, V)
    output = output.permute(1, 0, 3, 2, 4).reshape(B, NT * BT, H, V)
    output = output[:, :T].contiguous()

    final_state = state.reshape(B, H, K, V)
    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return output, final_state


@torch.inference_mode()
def chunk_gdn2_torch_gcu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int | None = 64,
    return_intermediate_states: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    disable_recompute: bool = False,
    cp_context=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the independently implemented chunked GDN2 inference path."""
    if _torch_gcu is None:
        raise RuntimeError("torch_gcu is required for chunk_gdn2_torch_gcu")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword arguments: {unknown}")
    if cp_context is not None:
        raise NotImplementedError("context parallelism is not supported")
    del disable_recompute

    validate_gdn2_torch_gcu_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states,
        chunk_indices=chunk_indices,
    )

    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = 64 if chunk_size is None else chunk_size

    q_work = q
    k_work = k
    if use_qk_l2norm_in_kernel:
        q_work = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
        k_work = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    gate = _activate_gate(
        g,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
    )

    if cu_seqlens is None:
        output, final_state = _run_chunked_batch(
            q_work,
            k_work,
            v,
            gate,
            b,
            w,
            scale=scale,
            initial_state=initial_state,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
            stable_scores=use_gate_in_kernel,
        )
    else:
        offsets_tensor = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        offsets = offsets_tensor.detach().cpu().tolist()
        if offsets[0] != 0 or offsets[-1] != q.shape[1]:
            raise ValueError(
                f"cu_seqlens must start at 0 and end at T={q.shape[1]}, got {offsets}"
            )
        if any(end < start for start, end in zip(offsets[:-1], offsets[1:])):
            raise ValueError(f"cu_seqlens must be nondecreasing, got {offsets}")

        output_parts = []
        final_state_parts = []
        for sequence_id, (start, end) in enumerate(
            zip(offsets[:-1], offsets[1:])
        ):
            if start == end:
                raise NotImplementedError("empty packed sequences are not supported")
            initial_state_i = (
                None
                if initial_state is None
                else initial_state[sequence_id : sequence_id + 1]
            )
            output_i, final_state_i = _run_chunked_batch(
                q_work[:, start:end],
                k_work[:, start:end],
                v[:, start:end],
                gate[:, start:end],
                b[:, start:end],
                w[:, start:end],
                scale=scale,
                initial_state=initial_state_i,
                state_v_first=state_v_first,
                chunk_size=chunk_size,
                stable_scores=use_gate_in_kernel,
            )
            output_parts.append(output_i)
            final_state_parts.append(final_state_i)
        output = torch.cat(output_parts, dim=1)
        final_state = torch.cat(final_state_parts, dim=0)

    return output, final_state if output_final_state else None
