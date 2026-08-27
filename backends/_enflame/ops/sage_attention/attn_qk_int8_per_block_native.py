# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SageAttention forward with a torch-gcu SDPA native path and Triton fallback.

The SageAttention inputs use INT8 storage with a scale for every 128 Q rows
and every 64 K rows.  The native Enflame FlashAttention interfaces currently
exposed through GCU Torch consume floating-point Q/K/V, so the ``torch-gcu``
backend performs an O(BHTD) dequantization with ordinary GCU Torch operators
and leaves the O(BHT^2D) attention work to the fused vendor kernel.  ``auto``
selects the native path when available and falls back to the FlagAttention
Triton SageAttention implementation otherwise (or for arbitrary masks).
"""

from functools import lru_cache
import importlib
import math
import os
import sys
from pathlib import Path

import torch

_Q_SCALE_BLOCK = 128
_K_SCALE_BLOCK = 64
_LN2 = math.log(2.0)
_BACKENDS = ("auto", "torch-gcu", "vllm-gcu", "flash-attn", "triton")

# The Triton SageAttention implementation lives in the FlagAttention repo.
# Default to the sibling FlagAttention checkout under the workspace root; set
# FLAGATTN_SAGE_ATTN_DIR to override (e.g. for a different container layout).
_FLAGATTN_OPS_DIR = Path(
    os.environ.get(
        "FLAGATTN_SAGE_ATTN_DIR",
        str(
            Path(__file__).resolve().parents[5]
            / "FlagAttention"
            / "src"
            / "flag_attn"
            / "runtime"
            / "backend"
            / "_enflame"
            / "sage_attention"
        ),
    )
)


class NativeBackendUnavailable(RuntimeError):
    """Raised when a requested Enflame native attention backend is missing."""


def _exception_text(exc):
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")


@lru_cache(maxsize=1)
def _load_torch_gcu_sdpa():
    try:
        importlib.import_module("torch_gcu")
        functional = importlib.import_module("torch.nn.functional")
        function = getattr(functional, "scaled_dot_product_attention")
        return function, ""
    except Exception as exc:  # torch_gcu can fail while loading TOPS libraries.
        return None, _exception_text(exc)


@lru_cache(maxsize=1)
def _load_triton_attention():
    """Load the FlagAttention Triton SageAttention implementation."""
    try:
        if not _FLAGATTN_OPS_DIR.is_dir():
            raise ImportError(f"FlagAttention ops dir not found: {_FLAGATTN_OPS_DIR}")
        if str(_FLAGATTN_OPS_DIR) not in sys.path:
            sys.path.insert(0, str(_FLAGATTN_OPS_DIR))
        module = importlib.import_module("attn_qk_int8_per_block")
        return module.forward, ""
    except Exception as exc:
        return None, _exception_text(exc)


@lru_cache(maxsize=1)
def _load_vllm_gcu_attention():
    """Probe for an optional vllm-gcu native attention backend.

    Not implemented yet; always reports unavailable so ``auto`` skips it.
    """
    try:
        raise ImportError("vllm-gcu native SageAttention backend is not implemented")
    except Exception as exc:
        return None, _exception_text(exc)


@lru_cache(maxsize=1)
def _load_flash_attention():
    """Probe for an optional standalone flash-attn backend.

    Not implemented yet; always reports unavailable so ``auto`` skips it.
    """
    try:
        raise ImportError("flash-attn native SageAttention backend is not implemented")
    except Exception as exc:
        return None, _exception_text(exc)


def native_backend_status():
    """Return availability and import diagnostics for every backend."""
    torch_function, torch_error = _load_torch_gcu_sdpa()
    vllm_function, vllm_error = _load_vllm_gcu_attention()
    flash_function, flash_error = _load_flash_attention()
    triton_function, triton_error = _load_triton_attention()
    return {
        "torch-gcu": {
            "available": torch_function is not None,
            "error": torch_error,
        },
        "vllm-gcu": {
            "available": vllm_function is not None,
            "error": vllm_error,
        },
        "flash-attn": {
            "available": flash_function is not None,
            "error": flash_error,
        },
        "triton": {
            "available": triton_function is not None,
            "error": triton_error,
        },
    }


def resolve_backend(
    backend="auto", attn_mask=None, return_lse=False, device=None
):
    """Resolve a requested backend, applying ``auto`` fallback to Triton."""
    if backend not in _BACKENDS:
        raise ValueError(
            f"backend must be one of {', '.join(_BACKENDS)}; got {backend!r}"
        )

    if device is None:
        device_type = None
    elif isinstance(device, str):
        device_type = device.split(":", 1)[0]
    else:
        device_type = device.type

    if backend == "triton":
        return "triton"

    if backend == "torch-gcu":
        if attn_mask is not None:
            raise NotImplementedError("torch-gcu does not support arbitrary attn_mask")
        if return_lse:
            raise NativeBackendUnavailable("torch-gcu SDPA does not expose LSE")
        torch_function, error = _load_torch_gcu_sdpa()
        if torch_function is None:
            raise NativeBackendUnavailable(
                "torch-gcu scaled_dot_product_attention is unavailable: " + error
            )
        return "torch-gcu"

    if backend == "vllm-gcu":
        function, error = _load_vllm_gcu_attention()
        if function is None:
            raise NativeBackendUnavailable(
                "vllm-gcu attention is unavailable: " + error
            )
        return "vllm-gcu"

    if backend == "flash-attn":
        function, error = _load_flash_attention()
        if function is None:
            raise NativeBackendUnavailable("flash-attn is unavailable: " + error)
        return "flash-attn"

    # backend == "auto"
    if attn_mask is not None:
        # torch-gcu does not support arbitrary masks; fall back to Triton.
        return "triton"

    if device_type not in (None, "gcu", "privateuseone"):
        # Native paths require a GCU device; fall back to Triton.
        return "triton"

    torch_function, _ = _load_torch_gcu_sdpa()
    if torch_function is not None and not return_lse:
        return "torch-gcu"

    vllm_function, _ = _load_vllm_gcu_attention()
    if vllm_function is not None:
        return "vllm-gcu"

    flash_function, _ = _load_flash_attention()
    if flash_function is not None:
        return "flash-attn"

    return "triton"


def _shape_metadata(q, k, v, q_scale, k_scale, tensor_layout, output_dtype):
    if tensor_layout == "HND":
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors")
        batch_size, num_query_heads, qo_len, head_dim = q.shape
        k_batch, num_kv_heads, kv_len, k_head_dim = k.shape
        expected_v_shape = (batch_size, num_kv_heads, kv_len, head_dim)
    elif tensor_layout == "NHD":
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors")
        batch_size, qo_len, num_query_heads, head_dim = q.shape
        k_batch, kv_len, num_kv_heads, k_head_dim = k.shape
        expected_v_shape = (batch_size, kv_len, num_kv_heads, head_dim)
    else:
        raise ValueError(f"tensor_layout {tensor_layout!r} is not supported")

    if q.dtype != torch.int8 or k.dtype != torch.int8:
        raise TypeError("native SageAttention expects INT8 q and k tensors")
    if v.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("v must have float16 or bfloat16 dtype")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("output_dtype must be torch.float16 or torch.bfloat16")
    if k_batch != batch_size or k_head_dim != head_dim:
        raise ValueError("q and k batch/head dimensions do not match")
    if tuple(v.shape) != expected_v_shape:
        raise ValueError(
            f"v has shape {tuple(v.shape)}, expected {expected_v_shape}"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("the number of KV heads must divide the number of Q heads")
    if head_dim not in (64, 128):
        raise ValueError("the Enflame native path supports head_dim 64 or 128")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q_scale.device != q.device or k_scale.device != q.device:
        raise ValueError("q_scale and k_scale must be on the same device as q")
    if not q_scale.is_floating_point() or not k_scale.is_floating_point():
        raise TypeError("q_scale and k_scale must be floating-point tensors")

    expected_q_scale = (
        batch_size,
        num_query_heads,
        (qo_len + _Q_SCALE_BLOCK - 1) // _Q_SCALE_BLOCK,
    )
    expected_k_scale = (
        batch_size,
        num_kv_heads,
        (kv_len + _K_SCALE_BLOCK - 1) // _K_SCALE_BLOCK,
    )
    if tuple(q_scale.shape) != expected_q_scale:
        raise ValueError(
            f"q_scale has shape {tuple(q_scale.shape)}, expected {expected_q_scale}"
        )
    if tuple(k_scale.shape) != expected_k_scale:
        raise ValueError(
            f"k_scale has shape {tuple(k_scale.shape)}, expected {expected_k_scale}"
        )
    return (
        batch_size,
        num_query_heads,
        num_kv_heads,
        qo_len,
        kv_len,
        head_dim,
    )


def _as_nhd(tensor, tensor_layout):
    if tensor_layout == "HND":
        return tensor.transpose(1, 2).contiguous()
    return tensor.contiguous()


def _dequantize_per_block(tensor_nhd, scale, block_size, scale_factor=1.0):
    """Dequantize an NHD tensor without materializing a full-size scale tensor."""
    batch_size, seq_len, num_heads, head_dim = tensor_nhd.shape
    tensor_fp16 = tensor_nhd.to(torch.float16)
    scale_fp16 = (scale * scale_factor).to(torch.float16)

    if seq_len % block_size == 0:
        num_blocks = seq_len // block_size
        blocked = tensor_fp16.reshape(
            batch_size, num_blocks, block_size, num_heads, head_dim
        )
        blocked.mul_(
            scale_fp16.permute(0, 2, 1)
            .reshape(batch_size, num_blocks, 1, num_heads, 1)
        )
        return tensor_fp16

    # Tail shapes are uncommon in the long-sequence performance suite.  This
    # path keeps them correct while the divisible path avoids an expanded scale.
    row_scale = scale_fp16.repeat_interleave(block_size, dim=-1)[..., :seq_len]
    tensor_fp16.mul_(row_scale.transpose(1, 2).unsqueeze(-1))
    return tensor_fp16


def _prepare_native_inputs(q, k, v, q_scale, k_scale, tensor_layout):
    q_nhd = _as_nhd(q, tensor_layout)
    k_nhd = _as_nhd(k, tensor_layout)
    v_nhd = _as_nhd(v, tensor_layout).to(torch.float16)

    # per_block_int8() folds log2(e) into q_scale because the Triton kernel
    # uses exp2.  Native FlashAttention uses exp, hence the ln(2) correction.
    q_fp16 = _dequantize_per_block(
        q_nhd, q_scale, _Q_SCALE_BLOCK, scale_factor=_LN2
    )
    k_fp16 = _dequantize_per_block(k_nhd, k_scale, _K_SCALE_BLOCK)
    return q_fp16, k_fp16, v_nhd


def _format_output(output_nhd, tensor_layout, output_dtype):
    output = output_nhd.to(output_dtype)
    if tensor_layout == "HND":
        return output.transpose(1, 2).contiguous()
    return output


def _forward_torch_gcu(q, k, v, return_lse):
    if return_lse:
        raise NativeBackendUnavailable("torch-gcu SDPA does not expose LSE")
    function, error = _load_torch_gcu_sdpa()
    if function is None:
        raise NativeBackendUnavailable(
            "torch-gcu scaled_dot_product_attention is unavailable: " + error
        )

    # PyTorch SDPA uses B,H,N,D while the vendor FlashAttention wrappers use
    # B,N,H,D.  The last (head-dimension) stride remains one after transpose,
    # which satisfies torch-gcu's FlashAttention selection constraint.
    q_hnd = q.transpose(1, 2)
    k_hnd = k.transpose(1, 2)
    v_hnd = v.transpose(1, 2)
    num_groups = q_hnd.shape[1] // k_hnd.shape[1]
    if num_groups == 1:
        output_hnd = function(
            q_hnd,
            k_hnd,
            v_hnd,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
    else:
        try:
            output_hnd = function(
                q_hnd,
                k_hnd,
                v_hnd,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
                enable_gqa=True,
            )
        except TypeError:
            # Older torch-gcu/PyTorch combinations predate enable_gqa.  This
            # compatibility path uses more bandwidth but retains correctness.
            k_hnd = k_hnd.repeat_interleave(num_groups, dim=1)
            v_hnd = v_hnd.repeat_interleave(num_groups, dim=1)
            output_hnd = function(
                q_hnd,
                k_hnd,
                v_hnd,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
    return output_hnd.transpose(1, 2), None


def _triton_forward(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    attn_mask=None,
    output_dtype=torch.float16,
    return_lse=False,
    maxnreg=None,
    block_m=None,
    num_warps=None,
    num_stages=None,
):
    """Delegate to the FlagAttention Triton SageAttention implementation."""
    triton_forward_impl, error = _load_triton_attention()
    if triton_forward_impl is None:
        raise NativeBackendUnavailable(
            "Triton SageAttention is unavailable: " + error
        )
    return triton_forward_impl(
        q,
        k,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        attn_mask=attn_mask,
        output_dtype=output_dtype,
        return_lse=return_lse,
        maxnreg=maxnreg,
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _forward_vllm_gcu(
    q, k, v, batch_size, num_query_heads, qo_len, kv_len, return_lse
):
    """Placeholder for an optional vllm-gcu native backend (not implemented)."""
    function, error = _load_vllm_gcu_attention()
    if function is None:
        raise NativeBackendUnavailable("vllm-gcu attention is unavailable: " + error)
    raise NotImplementedError("vllm-gcu native SageAttention is not implemented")


def _forward_flash_attention(q, k, v, return_lse):
    """Placeholder for an optional standalone flash-attn backend (not implemented)."""
    function, error = _load_flash_attention()
    if function is None:
        raise NativeBackendUnavailable("flash-attn is unavailable: " + error)
    raise NotImplementedError("flash-attn native SageAttention is not implemented")


def forward(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    attn_mask=None,
    output_dtype=torch.float16,
    return_lse=False,
    maxnreg=None,
    block_m=None,
    num_warps=None,
    num_stages=None,
    backend="auto",
):
    """Run SageAttention through an Enflame native or Triton backend.

    Native backends currently support dense, non-causal attention.  ``auto``
    preserves the full existing API by using Triton for arbitrary masks and
    when neither optional native package is installed.
    """
    selected_backend = resolve_backend(
        backend, attn_mask, return_lse, device=q.device
    )
    if selected_backend == "triton":
        return _triton_forward(
            q,
            k,
            v,
            q_scale,
            k_scale,
            tensor_layout=tensor_layout,
            attn_mask=attn_mask,
            output_dtype=output_dtype,
            return_lse=return_lse,
            maxnreg=maxnreg,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if attn_mask is not None:
        raise NotImplementedError(
            f"backend={selected_backend!r} does not support arbitrary attn_mask"
        )

    (
        batch_size,
        num_query_heads,
        _,
        qo_len,
        kv_len,
        _,
    ) = _shape_metadata(
        q, k, v, q_scale, k_scale, tensor_layout, output_dtype
    )
    q_native, k_native, v_native = _prepare_native_inputs(
        q, k, v, q_scale, k_scale, tensor_layout
    )

    if selected_backend == "torch-gcu":
        output_nhd, lse = _forward_torch_gcu(
            q_native, k_native, v_native, return_lse
        )
    elif selected_backend == "vllm-gcu":
        output_nhd, lse = _forward_vllm_gcu(
            q_native,
            k_native,
            v_native,
            batch_size,
            num_query_heads,
            qo_len,
            kv_len,
            return_lse,
        )
    else:
        output_nhd, lse = _forward_flash_attention(
            q_native, k_native, v_native, return_lse
        )

    output = _format_output(output_nhd, tensor_layout, output_dtype)
    if return_lse:
        lse = _format_lse(lse, batch_size, num_query_heads, qo_len)
    else:
        lse = torch.empty((0,), dtype=torch.float32, device="cpu")
    return output, lse
