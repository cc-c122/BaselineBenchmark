from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from chunk_gla_baseline_utils import select_safe_chunk_size


_THIS_DIR = Path(__file__).resolve().parent
_SOURCE = _THIS_DIR / "chunk_gla_tops_extension.cpp"
_BUILD_DIR = _THIS_DIR / "build_chunk_gla_tops"
_SITE_PACKAGES = Path(torch.__file__).resolve().parent.parent
_TORCH_GCU_ROOT = _SITE_PACKAGES / "torch_gcu"
_BUILD_DIR.mkdir(exist_ok=True)
_MASK_CACHE: dict[tuple[str, int], torch.Tensor] = {}

_EXTENSION = load(
    name="flaggems_chunk_gla_tops_v1",
    sources=[str(_SOURCE)],
    extra_include_paths=[
        str(_TORCH_GCU_ROOT / "include"),
        "/opt/tops/include",
        "/opt/tops/include/gcu",
    ],
    extra_cflags=["-O2"],
    extra_ldflags=[
        f"-L{_TORCH_GCU_ROOT / 'lib'}",
        "-ltorch_gcu",
        "-L/lib",
        "-ltopsaten",
        f"-Wl,-rpath,{_TORCH_GCU_ROOT / 'lib'}",
        "-Wl,-rpath,/lib",
    ],
    build_directory=str(_BUILD_DIR),
    verbose=False,
)


def _causal_mask(device: torch.device, length: int) -> torch.Tensor:
    key = (str(device), length)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        indices = torch.arange(length, device=device)
        mask = indices[:, None] >= indices[None, :]
        _MASK_CACHE[key] = mask
    return mask


def chunk_gla_tops(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward-only GLA implemented with native TopsAten C++ kernels."""
    if q.ndim != 4 or k.shape != q.shape or g.shape != q.shape:
        raise ValueError("q, k and g must have shape [B, T, H, K]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B, T, H, V]")
    if any(tensor.device != q.device for tensor in (k, v, g)):
        raise ValueError("q, k, v and g must be on the same device")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, g)):
        raise TypeError("q, k, v and g must all be bfloat16")
    if q.requires_grad or k.requires_grad or v.requires_grad or g.requires_grad:
        raise RuntimeError("the Tops C++ baseline is forward-only")

    requested_chunk_size = int(kwargs.pop("chunk_size", 128))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword arguments: {unexpected}")

    batch, sequence_length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("varlen chunk GLA expects q.shape[0] == 1")
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1:
            raise TypeError("cu_seqlens must be a rank-1 int32 tensor")
        boundaries = [int(value) for value in cu_seqlens.cpu().tolist()]
        if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != sequence_length:
            raise ValueError("cu_seqlens must start at 0 and end at T")
        if any(end <= begin for begin, end in zip(boundaries, boundaries[1:])):
            raise ValueError("cu_seqlens must be strictly increasing")
        state_count = len(boundaries) - 1
    else:
        boundaries = None
        state_count = batch

    expected_state_shape = (state_count, heads, key_dim, value_dim)
    if initial_state is not None:
        physical_shape = (
            (state_count, heads, value_dim, key_dim)
            if state_v_first
            else expected_state_shape
        )
        if tuple(initial_state.shape) != physical_shape:
            raise ValueError(f"initial_state must have shape {physical_shape}")
        if initial_state.dtype != torch.float32 or initial_state.device != q.device:
            raise TypeError("initial_state must be float32 on the input device")

    actual_scale = key_dim ** -0.5 if scale is None else float(scale)

    if boundaries is not None:
        outputs = []
        states = []
        for index, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
            state0 = None if initial_state is None else initial_state[index:index + 1]
            output, final_state = chunk_gla_tops(
                q[:, begin:end],
                k[:, begin:end],
                v[:, begin:end],
                g[:, begin:end],
                scale=actual_scale,
                initial_state=state0,
                output_final_state=output_final_state,
                state_v_first=state_v_first,
                chunk_size=requested_chunk_size,
            )
            outputs.append(output)
            if output_final_state:
                states.append(final_state)
        return (
            torch.cat(outputs, dim=1),
            torch.cat(states, dim=0) if output_final_state else None,
        )

    chunk_size = select_safe_chunk_size(g, requested_chunk_size)
    return _EXTENSION.forward(
        q,
        k,
        v,
        g,
        actual_scale,
        initial_state,
        _causal_mask(q.device, chunk_size),
        output_final_state,
        state_v_first,
        chunk_size,
    )


chunk_gla = chunk_gla_tops

__all__ = ["chunk_gla", "chunk_gla_tops"]
