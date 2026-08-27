"""Correctness tests for the experimental Torch GCU GDN2 implementation."""

import math
import sys
from pathlib import Path

import pytest
import torch

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from gdn2 import chunk_gdn2_torch_gcu


def _gcu_available() -> bool:
    return hasattr(torch, "gcu") and torch.gcu.is_available()


pytestmark = [
    pytest.mark.chunk_gdn2,
    pytest.mark.gdn2,
    pytest.mark.skipif(
        not _gcu_available(),
        reason="Torch GCU GDN2 tests require an Enflame GCU",
    ),
]


def _error_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).square().mean().sqrt().item()
    base = expected.float().square().mean().sqrt().item()
    return diff / (base + 1e-8)


def _run_recurrent_reference(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    scale,
    initial_state,
    state_v_first,
):
    """Independent float32 token recurrence for one equal-length batch."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        state = initial_state.transpose(-1, -2) if state_v_first else initial_state
        state = state.float().contiguous()

    output = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)
    for token_id in range(T):
        q_i = q[:, token_id].float() * scale
        k_i = k[:, token_id].float()
        v_i = v[:, token_id].float()
        g_i = g[:, token_id].float()
        b_i = b[:, token_id].float()
        w_i = w[:, token_id].float()

        state = state * torch.exp(g_i).unsqueeze(-1)
        erased_value = (
            (k_i * b_i).unsqueeze(-1) * state
        ).sum(dim=-2)
        value_update = v_i * w_i - erased_value
        state = state + torch.einsum("bhk,bhv->bhkv", k_i, value_update)
        output[:, token_id] = torch.einsum("bhk,bhkv->bhv", q_i, state)

    final_state = state.transpose(-1, -2).contiguous() if state_v_first else state
    return output.to(v.dtype), final_state


@torch.inference_mode()
def _recurrent_reference(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    scale,
    initial_state,
    state_v_first,
    cu_seqlens,
    cu_seqlens_cpu,
    **_kwargs,
):
    if cu_seqlens is None:
        return _run_recurrent_reference(
            q,
            k,
            v,
            g,
            b,
            w,
            scale=scale,
            initial_state=initial_state,
            state_v_first=state_v_first,
        )

    offsets_tensor = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    offsets = offsets_tensor.detach().cpu().tolist()
    output_parts = []
    final_state_parts = []
    for sequence_id, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        initial_state_i = (
            None
            if initial_state is None
            else initial_state[sequence_id : sequence_id + 1]
        )
        output_i, final_state_i = _run_recurrent_reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            b[:, start:end],
            w[:, start:end],
            scale=scale,
            initial_state=initial_state_i,
            state_v_first=state_v_first,
        )
        output_parts.append(output_i)
        final_state_parts.append(final_state_i)
    return torch.cat(output_parts, dim=1), torch.cat(final_state_parts, dim=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_lens", [[64], [29, 35, 17]])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("state_v_first", [False, True])
@torch.inference_mode()
def test_chunk_gdn2_torch_gcu_matches_reference(
    dtype,
    seq_lens,
    use_initial_state,
    state_v_first,
):
    torch.manual_seed(42)

    device = torch.device("gcu")
    T = sum(seq_lens)
    N = len(seq_lens)
    H = 2
    K = 64
    V = 64
    q = torch.randn(1, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(1, T, H, K, device=device) * 0.1).to(dtype)
    b = torch.rand(1, T, H, K, device=device, dtype=dtype)
    w = torch.rand(1, T, H, V, device=device, dtype=dtype)

    initial_state = None
    if use_initial_state:
        state_shape = (N, H, V, K) if state_v_first else (N, H, K, V)
        initial_state = 0.01 * torch.randn(
            *state_shape,
            device=device,
            dtype=torch.float32,
        )

    cu_seqlens = None
    cu_seqlens_cpu = None
    if N > 1:
        offsets = [0]
        for length in seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int32)
        cu_seqlens = cu_seqlens_cpu.to(device)

    kwargs = {
        "scale": K**-0.5,
        "initial_state": initial_state,
        "output_final_state": True,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_cpu": cu_seqlens_cpu,
        "chunk_indices": None,
        "chunk_size": 64,
        "safe_gate": False,
        "lower_bound": None,
        "use_gate_in_kernel": False,
        "A_log": None,
        "dt_bias": None,
        "disable_recompute": True,
        "return_intermediate_states": False,
        "state_v_first": state_v_first,
    }

    actual, actual_state = chunk_gdn2_torch_gcu(
        q,
        k,
        v,
        g,
        b,
        w,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    )
    expected, expected_state = _recurrent_reference(q, k, v, g, b, w, **kwargs)

    output_ratio = _error_ratio(actual, expected)
    state_ratio = _error_ratio(actual_state, expected_state)
    assert output_ratio < 0.01, f"output error ratio: {output_ratio}"
    assert state_ratio < 0.01, f"state error ratio: {state_ratio}"


@torch.inference_mode()
def test_chunk_gdn2_torch_gcu_state_layout_equivalence():
    """V-first and K-first inputs must represent the same recurrent state."""
    torch.manual_seed(43)

    device = torch.device("gcu")
    B, T, H, K, V = 1, 17, 2, 32, 48
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, K, device=device) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)

    initial_state_vk = 0.01 * torch.randn(
        B,
        H,
        V,
        K,
        device=device,
        dtype=torch.float32,
    )
    initial_state_kv = initial_state_vk.transpose(-1, -2).contiguous()
    kwargs = {
        "scale": K**-0.5,
        "output_final_state": True,
        "chunk_size": 64,
        "safe_gate": False,
        "lower_bound": None,
        "use_gate_in_kernel": False,
        "A_log": None,
        "dt_bias": None,
        "return_intermediate_states": False,
    }

    output_vk, final_state_vk = chunk_gdn2_torch_gcu(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state=initial_state_vk,
        state_v_first=True,
        **kwargs,
    )
    output_kv, final_state_kv = chunk_gdn2_torch_gcu(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state=initial_state_kv,
        state_v_first=False,
        **kwargs,
    )

    output_ratio = _error_ratio(output_vk, output_kv)
    state_ratio = _error_ratio(
        final_state_vk,
        final_state_kv.transpose(-1, -2).contiguous(),
    )
    assert output_ratio < 1e-6, f"layout output error ratio: {output_ratio}"
    assert state_ratio < 1e-6, f"layout state error ratio: {state_ratio}"
