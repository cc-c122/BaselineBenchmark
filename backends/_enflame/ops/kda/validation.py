"""Input validation for the experimental Torch GCU KDA path."""

from __future__ import annotations

import torch


def validate_kda_torch_gcu_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    state_v_first: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: float | None,
    chunk_size: int | None,
) -> None:
    inputs = {"q": q, "k": k, "v": v, "g": g, "beta": beta}

    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        raise RuntimeError("chunk_kda_torch_gcu requires an available Torch GCU")
    if q.device.type != "gcu":
        raise ValueError(f"chunk_kda_torch_gcu requires GCU inputs, got {q.device}")
    if any(tensor.device != q.device for tensor in inputs.values()):
        raise ValueError("q, k, v, g, and beta must be on the same GCU device")
    if any(tensor.dtype != torch.bfloat16 for tensor in inputs.values()):
        actual = ", ".join(f"{name}={tensor.dtype}" for name, tensor in inputs.items())
        raise ValueError(f"Torch GCU KDA currently requires bfloat16 inputs, got {actual}")
    if any(not tensor.is_contiguous() for tensor in inputs.values()):
        raise ValueError("Torch GCU KDA currently requires contiguous inputs")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4:
        raise ValueError("q, k, v, and g must have rank 4")
    if beta.ndim != 3:
        raise ValueError("beta must have rank 3")

    B, T, H, K = q.shape
    if T == 0:
        raise ValueError("Torch GCU KDA currently requires a non-empty sequence")
    if K != 128 or v.shape[-1] != 128:
        raise NotImplementedError(
            f"Torch GCU KDA currently supports K=V=128, got K={K}, V={v.shape[-1]}"
        )
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        raise NotImplementedError(
            "Torch GCU KDA currently requires q, k, v, and g shape [B, T, H, 128]"
        )
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must have shape {(B, T, H)}, got {tuple(beta.shape)}")
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError("use_qk_l2norm_in_kernel=False is not supported")
    if not use_gate_in_kernel:
        raise NotImplementedError("use_gate_in_kernel=False is not supported")
    if not use_beta_sigmoid_in_kernel:
        raise NotImplementedError("use_beta_sigmoid_in_kernel=False is not supported")
    if allow_neg_eigval:
        raise NotImplementedError("allow_neg_eigval=True is not supported")
    if not safe_gate:
        raise NotImplementedError("safe_gate=False is not supported")
    if lower_bound is None or not -5 <= lower_bound < 0:
        raise ValueError(f"lower_bound must satisfy -5 <= lower_bound < 0, got {lower_bound}")
    if chunk_size not in (None, 16, 32):
        raise NotImplementedError(f"chunk_size must be 16, 32, or None, got {chunk_size}")

    if A_log is None or A_log.dtype != torch.float32 or A_log.shape != (H,):
        actual = None if A_log is None else (tuple(A_log.shape), A_log.dtype)
        raise ValueError(f"A_log must be float32 with shape {(H,)}, got {actual}")
    if dt_bias is None or dt_bias.dtype != torch.float32 or dt_bias.shape != (H, K):
        actual = None if dt_bias is None else (tuple(dt_bias.shape), dt_bias.dtype)
        raise ValueError(f"dt_bias must be float32 with shape {(H, K)}, got {actual}")
    if A_log.device != q.device or dt_bias.device != q.device:
        raise ValueError("A_log and dt_bias must be on the input GCU device")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("A_log and dt_bias must be contiguous")

    N = B
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("packed varlen input requires B=1")
        if (
            cu_seqlens.device != q.device
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.ndim != 1
        ):
            raise ValueError("cu_seqlens must be a 1D int32 tensor on the input device")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least two offsets")
        N = cu_seqlens.numel() - 1

    if initial_state is not None:
        V = v.shape[-1]
        expected = (N, H, V, K) if state_v_first else (N, H, K, V)
        if initial_state.dtype != torch.float32 or initial_state.shape != expected:
            raise ValueError(
                f"initial_state must be float32 with shape {expected}, "
                f"got shape={tuple(initial_state.shape)}, dtype={initial_state.dtype}"
            )
        if initial_state.device != q.device or not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous on the input GCU device")
