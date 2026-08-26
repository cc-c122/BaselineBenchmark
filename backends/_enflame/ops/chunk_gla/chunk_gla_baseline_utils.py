from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# The midpoint factorization forms exp(+span/2) in FP32. Keeping the full
# cumulative span below 144 leaves about 16 natural-log units of headroom to
# FP32 overflow (log(max_float) ~= 88.7), including accumulation and ordinary
# activation magnitudes.
MAX_CUMULATIVE_GATE_SPAN = 144.0
MAX_CHUNK_SIZE = 128


def select_safe_chunk_size(g: torch.Tensor, requested: int) -> int:
    """Choose the largest numerically safe aligned chunk for log forget gates.

    GLA receives log-space forget gates, so every gate must be non-positive.
    For one aligned chunk, ``-sum(g)`` bounds the range of its cumulative gate.
    The midpoint factorization then bounds each positive exponent by half that
    range. The check is data-dependent and deliberately included in benchmark
    timing; correctness must not depend on benchmark-only assumptions.
    """
    if requested < 1:
        raise ValueError("chunk_size must be positive")
    if g.numel() == 0:
        return 1

    seqlen = g.shape[1]
    candidate = min(int(requested), MAX_CHUNK_SIZE, seqlen)
    # Matrix kernels are most predictable at power-of-two block sizes.
    candidate = 1 << int(math.floor(math.log2(candidate)))
    gate = g.detach().float()
    checked_sign = False

    while candidate > 1:
        padding = (-seqlen) % candidate
        padded = F.pad(gate, (0, 0, 0, 0, 0, padding)) if padding else gate
        block_span = (
            -padded.reshape(
                g.shape[0], -1, candidate, g.shape[2], g.shape[3]
            ).sum(dim=2)
        ).amax()

        if not checked_sign:
            statistics = torch.stack((gate.amax(), block_span)).cpu()
            maximum_gate = float(statistics[0])
            maximum_span = float(statistics[1])
            checked_sign = True
            if maximum_gate > 0.0:
                raise ValueError(
                    "chunk GLA expects non-positive log forget gates; "
                    f"found max(g)={maximum_gate}"
                )
        else:
            maximum_span = float(block_span.cpu())

        if maximum_span <= MAX_CUMULATIVE_GATE_SPAN:
            return candidate
        candidate //= 2

    return 1


__all__ = [
    "MAX_CHUNK_SIZE",
    "MAX_CUMULATIVE_GATE_SPAN",
    "select_safe_chunk_size",
]
