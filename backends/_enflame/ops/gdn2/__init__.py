"""Experimental GDN2 implementation backed by Torch GCU operators.

This package is intentionally isolated from the production Enflame dispatch.
It provides an inference-only Torch implementation for correctness and
performance experiments on S60.
"""

from .forward import chunk_gdn2_torch_gcu

__all__ = ["chunk_gdn2_torch_gcu"]
