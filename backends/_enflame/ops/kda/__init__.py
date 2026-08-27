"""Experimental KDA implementation backed by Torch GCU operators.

This package is intentionally not re-exported by the Enflame backend.  It is
kept separate from the production Triton implementation while its semantics
and device coverage are validated on S60.
"""

from .forward import chunk_kda_torch_gcu

__all__ = ["chunk_kda_torch_gcu"]
