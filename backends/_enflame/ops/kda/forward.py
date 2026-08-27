"""Chunked KDA forward implemented with Torch GCU operators.

This remains an isolated, inference-only experiment: it is intentionally not
registered as the production KDA implementation. Unlike the original
correctness-first version, the hot path follows the same two-stage chunk
algorithm as the S60 Triton implementation. K1 prepares all chunks in
parallel and K2 keeps only the inter-chunk state recurrence serial.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import torch_gcu as _torch_gcu
except ImportError:
    _torch_gcu = None

from .validation import validate_kda_torch_gcu_inputs


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


def _pad_h_major_beta(
    beta: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Convert [B, T, H] to contiguous [B*H, NT, BT]."""
    B, T, H = beta.shape
    NT = (T + chunk_size - 1) // chunk_size
    padded_t = NT * chunk_size
    beta = beta.permute(0, 2, 1).contiguous().reshape(B * H, T)
    if padded_t != T:
        beta = F.pad(beta, (0, padded_t - T))
    return beta.reshape(B * H, NT, chunk_size)


def _triangular_inverse(
    lower: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Compute (I + L)^-1 for a strictly-lower nilpotent matrix L.

    This is the finite product used by the strict S60 Triton K1 path:
    (I-L)(I+L^2)(I+L^4)... . FP16 is intentional here; the existing S60
    path also uses FP16 operands for these small triangular matrix products.
    """
    matrix_count = lower.shape[0]
    lower_power = lower.to(torch.float16)
    identity = torch.eye(
        chunk_size,
        dtype=torch.float16,
        device=lower.device,
    ).expand(matrix_count, -1, -1)
    inverse = identity - lower_power

    power = 2
    while power < chunk_size:
        lower_power = torch.bmm(lower_power, lower_power)
        inverse = inverse + torch.bmm(inverse, lower_power)
        power *= 2
    return inverse.to(torch.bfloat16)


def _build_chunk_workspaces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run K1 and return chunk-major workspaces consumed by K2.

    Returned shapes are:
      wq:          [NT, B*H, 2*BT, K]
      out_state_a: [NT, B*H, BT+K, BT]
      Akk_inv:     [NT, B*H, BT, BT]
      beta_v:      [NT, B*H, BT, V]
      state_decay: [NT, B*H, K]
    """
    B, T, H, K = q.shape
    BH = B * H
    BT = chunk_size
    NT = (T + BT - 1) // BT
    matrix_count = BH * NT

    q_chunks = _pad_h_major_chunks(q, chunk_size=BT)
    k_chunks = _pad_h_major_chunks(k, chunk_size=BT)
    v_chunks = _pad_h_major_chunks(v, chunk_size=BT)
    g_chunks = _pad_h_major_chunks(g, chunk_size=BT)
    beta_chunks = _pad_h_major_beta(beta, chunk_size=BT)

    g_cumsum = torch.cumsum(g_chunks, dim=2)
    g_last = g_cumsum[:, :, -1]

    gate_from_start = torch.exp(g_cumsum)
    qg = (q_chunks * gate_from_start).to(torch.bfloat16)
    w = (k_chunks * beta_chunks.unsqueeze(-1) * gate_from_start).to(
        torch.bfloat16
    )
    del gate_from_start
    kg = (k_chunks * torch.exp(g_last.unsqueeze(2) - g_cumsum)).to(
        torch.bfloat16
    )

    # Center the two QK operands at the last row of the first 16-row half.
    # Every retained triangular product then stays inside the FP32/BF16
    # exponent range. The invalid bottom-left block is never formed.
    split = BT // 2
    shift = g_cumsum[:, :, split - 1 : split]
    k_left = (k_chunks * torch.exp(shift - g_cumsum)).to(torch.bfloat16)
    gate_from_shift = torch.exp(g_cumsum - shift)
    q_right = (q_chunks * gate_from_shift).to(torch.bfloat16)
    k_right = (k_chunks * gate_from_shift).to(torch.bfloat16)
    del gate_from_shift

    # Form only the three retained 16x16 block pairs. The previous eager
    # implementation zero-filled full 32x64 operands, so the matrix engine
    # still performed four times as many multiply-adds. S60 benchmarks show
    # that the real 16x32 BMM shape is faster despite its smaller M dimension.
    k_left_low = k_left[:, :, :split].reshape(matrix_count, split, K)
    k_left_high = k_left[:, :, split:].reshape(matrix_count, split, K)
    q_right_low = q_right[:, :, :split].reshape(matrix_count, split, K)
    q_right_high = q_right[:, :, split:].reshape(matrix_count, split, K)
    k_right_low = k_right[:, :, :split].reshape(matrix_count, split, K)
    k_right_high = k_right[:, :, split:].reshape(matrix_count, split, K)

    right_low = torch.cat(
        (q_right_low.transpose(1, 2), k_right_low.transpose(1, 2)),
        dim=2,
    )
    right_high = torch.cat(
        (q_right_high.transpose(1, 2), k_right_high.transpose(1, 2)),
        dim=2,
    )
    pair_low_low = torch.bmm(k_left_low, right_low).float()
    pair_low_high = torch.bmm(k_left_low, right_high).float()
    pair_high_high = torch.bmm(k_left_high, right_high).float()
    Aqk_ll_t, Akk_ll_t = torch.split(pair_low_low, split, dim=2)
    Aqk_lh_t, Akk_lh_t = torch.split(pair_low_high, split, dim=2)
    Aqk_hh_t, Akk_hh_t = torch.split(pair_high_high, split, dim=2)

    zero_block = torch.zeros_like(Aqk_ll_t)
    Aqk = torch.cat(
        (
            torch.cat((Aqk_ll_t.transpose(1, 2), zero_block), dim=2),
            torch.cat(
                (Aqk_lh_t.transpose(1, 2), Aqk_hh_t.transpose(1, 2)),
                dim=2,
            ),
        ),
        dim=1,
    )
    Akk = torch.cat(
        (
            torch.cat((Akk_ll_t.transpose(1, 2), zero_block), dim=2),
            torch.cat(
                (Akk_lh_t.transpose(1, 2), Akk_hh_t.transpose(1, 2)),
                dim=2,
            ),
        ),
        dim=1,
    )

    matrix_ids = torch.arange(BT, device=q.device, dtype=torch.int32)
    matrix_rows = matrix_ids.reshape(1, BT, 1)
    matrix_cols = matrix_ids.reshape(1, 1, BT)
    Aqk = torch.where(matrix_rows >= matrix_cols, Aqk * scale, 0.0)
    beta_flat = beta_chunks.reshape(matrix_count, BT)
    Akk = torch.where(
        matrix_rows > matrix_cols,
        Akk * beta_flat.unsqueeze(-1),
        0.0,
    )
    Aqk = Aqk.to(torch.bfloat16).reshape(BH, NT, BT, BT)
    Akk_inverse = _triangular_inverse(
        Akk,
        chunk_size=BT,
    ).reshape(BH, NT, BT, BT)

    beta_v = (v_chunks.float() * beta_chunks.unsqueeze(-1)).to(
        torch.bfloat16
    )

    # Materialize chunk-major, contiguous operands once. This avoids a
    # transpose/cat/copy in every serial K2 iteration.
    wq = torch.cat((w, qg), dim=2).permute(1, 0, 2, 3).contiguous()
    out_state_a = (
        torch.cat((Aqk, kg.transpose(-1, -2)), dim=2)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    Akk_inverse = Akk_inverse.permute(1, 0, 2, 3).contiguous()
    beta_v = beta_v.permute(1, 0, 2, 3).contiguous()
    state_decay = torch.exp(g_last).permute(1, 0, 2).contiguous()
    return wq, out_state_a, Akk_inverse, beta_v, state_decay


def _run_chunked_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run equal-length sequences with one BH-batched chunk recurrence."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    BT = chunk_size
    NT = (T + BT - 1) // BT

    if T == 0:
        output = torch.empty(
            B,
            0,
            H,
            V,
            dtype=v.dtype,
            device=v.device,
        )
        final_state = _initial_state_kv(
            initial_state,
            B=B,
            H=H,
            K=K,
            V=V,
            device=q.device,
            state_v_first=state_v_first,
        ).reshape(B, H, K, V)
        if state_v_first:
            final_state = final_state.transpose(-1, -2).contiguous()
        return output, final_state

    wq, out_state_a, Akk_inverse, beta_v, state_decay = (
        _build_chunk_workspaces(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            chunk_size=BT,
        )
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
        dtype=torch.bfloat16,
        device=q.device,
    )

    for chunk_id in range(NT):
        state_bf16 = state.to(torch.bfloat16)
        state_projection = torch.bmm(
            wq[chunk_id],
            state_bf16,
        ).float()
        kh = state_projection[:, :BT]
        qh = state_projection[:, BT:]

        residual = (beta_v[chunk_id].float() - kh).to(torch.bfloat16)
        v_new = torch.bmm(
            Akk_inverse[chunk_id],
            residual,
        )

        output_state_update = torch.bmm(
            out_state_a[chunk_id],
            v_new,
        ).float()
        output_correction = output_state_update[:, :BT]
        state_update = output_state_update[:, BT:]
        output_chunks[chunk_id] = (scale * qh + output_correction).to(
            torch.bfloat16
        )
        state = state * state_decay[chunk_id].unsqueeze(-1) + state_update

    output = output_chunks.reshape(NT, B, H, BT, V)
    output = output.permute(1, 0, 3, 2, 4).reshape(B, NT * BT, H, V)
    output = output[:, :T].contiguous()

    final_state = state.reshape(B, H, K, V)
    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return output, final_state


@torch.inference_mode()
def chunk_kda_torch_gcu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = -5.0,
    state_v_first: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run chunked KDA inference with standard Torch GCU operators."""
    if _torch_gcu is None:
        raise RuntimeError("torch_gcu is required for chunk_kda_torch_gcu")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword arguments: {unknown}")

    validate_kda_torch_gcu_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
    )

    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = 32 if chunk_size is None else chunk_size

    qf = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
    kf = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    gf = g.float() + dt_bias.reshape(1, 1, q.shape[2], q.shape[3])
    gf = lower_bound * torch.sigmoid(
        torch.exp(A_log).reshape(1, 1, q.shape[2], 1) * gf
    )
    betaf = torch.sigmoid(beta.float())

    if cu_seqlens is None:
        output, final_state = _run_chunked_batch(
            qf,
            kf,
            v,
            gf,
            betaf,
            scale=scale,
            initial_state=initial_state,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )
    else:
        offsets = cu_seqlens.detach().cpu().tolist()
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
            initial_state_i = (
                None
                if initial_state is None
                else initial_state[sequence_id : sequence_id + 1]
            )
            output_i, final_state_i = _run_chunked_batch(
                qf[:, start:end],
                kf[:, start:end],
                v[:, start:end],
                gf[:, start:end],
                betaf[:, start:end],
                scale=scale,
                initial_state=initial_state_i,
                state_v_first=state_v_first,
                chunk_size=chunk_size,
            )
            output_parts.append(output_i)
            final_state_parts.append(final_state_i)
        output = torch.cat(output_parts, dim=1)
        final_state = torch.cat(final_state_parts, dim=0)

    return output, final_state if output_final_state else None
