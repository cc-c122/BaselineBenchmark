"""Input validation for the experimental Torch GCU GDN2 path."""

from __future__ import annotations

import torch


def validate_gdn2_torch_gcu_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    state_v_first: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: float | None,
    chunk_size: int | None,
    return_intermediate_states: bool,
    chunk_indices: torch.Tensor | None,
) -> None:
    inputs = {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w}

    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        raise RuntimeError("chunk_gdn2_torch_gcu requires an available Torch GCU")
    if q.device.type != "gcu":
        raise ValueError(f"chunk_gdn2_torch_gcu requires GCU inputs, got {q.device}")
    if any(tensor.device != q.device for tensor in inputs.values()):
        raise ValueError("q, k, v, g, b, and w must be on the same GCU device")
    supported_dtypes = (torch.float16, torch.bfloat16)
    if any(tensor.dtype not in supported_dtypes for tensor in inputs.values()):
        actual = ", ".join(f"{name}={tensor.dtype}" for name, tensor in inputs.items())
        raise ValueError(
            "Torch GCU GDN2 currently requires float16 or bfloat16 inputs, "
            f"got {actual}"
        )
    if any(tensor.dtype != q.dtype for tensor in inputs.values()):
        raise ValueError("q, k, v, g, b, and w must have the same dtype")
    if any(not tensor.is_contiguous() for tensor in inputs.values()):
        raise ValueError("Torch GCU GDN2 currently requires contiguous inputs")
    if any(tensor.ndim != 4 for tensor in inputs.values()):
        raise ValueError("q, k, v, g, b, and w must have rank 4")

    B, T, H, K = q.shape
    V = v.shape[-1]
    if T == 0:
        raise ValueError("Torch GCU GDN2 currently requires a non-empty sequence")
    if K > 256:
        raise NotImplementedError(
            f"Torch GCU GDN2 currently supports K <= 256, got K={K}"
        )
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g, and b must have the same [B, T, H, K] shape")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            "v and w must have the same [B, T, H, V] shape and match q on [B, T, H]"
        )
    if chunk_size not in (None, 64):
        raise NotImplementedError(f"chunk_size must be 64 or None, got {chunk_size}")
    if return_intermediate_states:
        raise NotImplementedError("return_intermediate_states=True is not supported")
    if chunk_indices is not None:
        raise NotImplementedError("explicit chunk_indices are not supported")

    if use_gate_in_kernel:
        if A_log is None or A_log.dtype != torch.float32 or A_log.shape != (H,):
            actual = None if A_log is None else (tuple(A_log.shape), A_log.dtype)
            raise ValueError(f"A_log must be float32 with shape {(H,)}, got {actual}")
        if A_log.device != q.device or not A_log.is_contiguous():
            raise ValueError("A_log must be contiguous on the input GCU device")
        if dt_bias is not None:
            if dt_bias.dtype != torch.float32 or dt_bias.shape != (H, K):
                actual = (tuple(dt_bias.shape), dt_bias.dtype)
                raise ValueError(
                    f"dt_bias must be float32 with shape {(H, K)}, got {actual}"
                )
            if dt_bias.device != q.device or not dt_bias.is_contiguous():
                raise ValueError("dt_bias must be contiguous on the input GCU device")
        if safe_gate and (lower_bound is None or not -5 <= lower_bound < 0):
            raise ValueError(
                "safe gated GDN2 requires lower_bound in [-5, 0), "
                f"got {lower_bound}"
            )
    elif A_log is not None or dt_bias is not None:
        raise ValueError("A_log and dt_bias require use_gate_in_kernel=True")

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
    if cu_seqlens_cpu is not None:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens_cpu requires cu_seqlens")
        if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.ndim != 1:
            raise ValueError("cu_seqlens_cpu must be a 1D CPU tensor")
        if cu_seqlens_cpu.numel() != cu_seqlens.numel():
            raise ValueError("cu_seqlens_cpu and cu_seqlens must have the same length")

    if initial_state is not None:
        expected = (N, H, V, K) if state_v_first else (N, H, K, V)
        if initial_state.dtype != torch.float32 or initial_state.shape != expected:
            raise ValueError(
                f"initial_state must be float32 with shape {expected}, "
                f"got shape={tuple(initial_state.shape)}, dtype={initial_state.dtype}"
            )
        if initial_state.device != q.device or not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous on the input GCU device")
