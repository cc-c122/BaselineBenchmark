from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


OPS_DIR = Path(__file__).resolve().parents[1] / "ops" / "chunk_gla"
sys.path.insert(0, str(OPS_DIR))

from chunk_gla_baseline_utils import select_safe_chunk_size
from chunk_gla_tops import chunk_gla_tops


def _device() -> torch.device:
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("GCU is unavailable")
    return torch.device("gcu")


def _fp64_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent token-by-token CPU reference for the defining recurrence."""
    qd, kd, vd, gd = (tensor.double().cpu() for tensor in (q, k, v, g))
    batch, seqlen, heads, key_dim = qd.shape
    value_dim = vd.shape[-1]
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float64)
        if initial_state is None
        else initial_state.double().cpu().clone()
    )
    output = torch.empty(batch, seqlen, heads, value_dim, dtype=torch.float64)
    for token in range(seqlen):
        query = qd[:, token]
        key = kd[:, token]
        value = vd[:, token]
        log_gate = gd[:, token]
        state = (
            state * torch.exp(log_gate).unsqueeze(-1)
            + key.unsqueeze(-1) * value.unsqueeze(-2)
        )
        output[:, token] = (
            torch.einsum("bhk,bhkv->bhv", query, state) * scale
        )
    return output, state


@pytest.mark.parametrize("gate_mode,seqlen", [("typical", 257), ("negative4", 129)])
def test_matches_independent_fp64_recurrence(gate_mode, seqlen):
    device = _device()
    torch.manual_seed(101)
    shape = (1, seqlen, 2, 16)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(1, seqlen, 2, 13, dtype=torch.bfloat16)
    if gate_mode == "typical":
        g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    else:
        g_cpu = torch.full_like(q_cpu, -4.0)
    q, k, v, g = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu))
    scale = 1.0 / math.sqrt(shape[-1])

    actual, actual_state = chunk_gla_tops(
        q, k, v, g, scale=scale, output_final_state=True, chunk_size=256
    )
    expected, expected_state = _fp64_recurrence(
        q_cpu, k_cpu, v_cpu, g_cpu, scale
    )

    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_state).all()
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.float(), rtol=3e-4, atol=2e-5
    )


def test_safe_chunk_selector_reduces_extreme_gate_span():
    g = torch.full((1, 128, 1, 16), -4.0, device=_device(),
                   dtype=torch.bfloat16)
    assert select_safe_chunk_size(g, 256) == 32


def test_positive_log_gate_is_rejected():
    device = _device()
    q = torch.ones(1, 4, 1, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="non-positive log forget gates"):
        chunk_gla_tops(q, q, q, torch.ones_like(q), chunk_size=128)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_initial_and_final_state_match_fp64(state_v_first):
    device = _device()
    torch.manual_seed(131)
    batch, seqlen, heads, key_dim, value_dim = 2, 33, 2, 16, 13
    q_cpu = torch.randn(batch, seqlen, heads, key_dim, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(batch, seqlen, heads, value_dim, dtype=torch.bfloat16)
    g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    state_k_first = torch.randn(
        batch, heads, key_dim, value_dim, dtype=torch.float32
    )
    initial = (
        state_k_first.transpose(-1, -2).contiguous()
        if state_v_first
        else state_k_first
    )
    q, k, v, g, initial_gcu = (
        tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu, initial)
    )
    scale = 1.0 / math.sqrt(key_dim)

    actual, actual_state = chunk_gla_tops(
        q, k, v, g, scale=scale, initial_state=initial_gcu,
        output_final_state=True, state_v_first=state_v_first, chunk_size=128,
    )
    expected, expected_state = _fp64_recurrence(
        q_cpu, k_cpu, v_cpu, g_cpu, scale, state_k_first
    )
    if state_v_first:
        expected_state = expected_state.transpose(-1, -2).contiguous()

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.float(), rtol=3e-4, atol=2e-5
    )


def test_varlen_matches_independent_sequences():
    device = _device()
    torch.manual_seed(151)
    lengths = [7, 11, 5]
    seqlen = sum(lengths)
    q_cpu = torch.randn(1, seqlen, 2, 16, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(1, seqlen, 2, 13, dtype=torch.bfloat16)
    g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    q, k, v, g = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu))
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, device=device, dtype=torch.int32)
    scale = 0.25

    actual, actual_states = chunk_gla_tops(
        q, k, v, g, scale=scale, cu_seqlens=cu_seqlens,
        output_final_state=True, chunk_size=128,
    )
    expected_outputs = []
    expected_states = []
    for begin, end in zip(boundaries, boundaries[1:]):
        output, state = _fp64_recurrence(
            q_cpu[:, begin:end], k_cpu[:, begin:end], v_cpu[:, begin:end],
            g_cpu[:, begin:end], scale,
        )
        expected_outputs.append(output)
        expected_states.append(state)
    expected = torch.cat(expected_outputs, dim=1)
    expected_states = torch.cat(expected_states, dim=0)

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_states.cpu(), expected_states.float(), rtol=3e-4, atol=2e-5
    )


def test_wide_key_dispatch_matches_independent_fp64_recurrence():
    device = _device()
    torch.manual_seed(181)
    batch, seqlen, heads, key_dim, value_dim = 1, 17, 1, 128, 13
    q_cpu = torch.randn(
        batch, seqlen, heads, key_dim, dtype=torch.bfloat16
    )
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(
        batch, seqlen, heads, value_dim, dtype=torch.bfloat16
    )
    g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    q, k, v, g = (
        tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu)
    )
    scale = key_dim ** -0.5

    actual, actual_state = chunk_gla_tops(
        q, k, v, g, scale=scale, output_final_state=True,
        chunk_size=128,
    )
    expected, expected_state = _fp64_recurrence(
        q_cpu, k_cpu, v_cpu, g_cpu, scale
    )

    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_state).all()
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.float(),
        rtol=3e-4, atol=2e-5,
    )
