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

"""Vectorized torch_gcu reference for the Enflame sparse MLA operator."""

from typing import Optional, Tuple

import torch


@torch.inference_mode()
def flash_mla_sparse_fwd_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    query_chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse MLA using batched gather, QK, softmax and PV operations."""
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert indices.dtype == torch.int32
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape
    _, _, TOPK = indices.shape
    assert HKV == 1
    assert HQ in (64, 128)
    assert DQK in (512, 576)
    assert d_v == 512
    assert query_chunk_size > 0

    kv_flat = kv[:, 0, :]
    output = torch.empty((SQ, HQ, d_v), dtype=q.dtype, device=q.device)
    max_logits = torch.empty(
        (SQ, HQ), dtype=torch.float32, device=q.device
    )
    lse = torch.empty_like(max_logits)
    topk_offsets = torch.arange(
        TOPK, dtype=torch.int32, device=q.device
    )

    for start in range(0, SQ, query_chunk_size):
        end = min(start + query_chunk_size, SQ)
        chunk_size = end - start
        chunk_indices = indices[start:end, 0, :]
        valid = (chunk_indices >= 0) & (chunk_indices < SKV)
        if topk_length is not None:
            valid &= topk_offsets[None, :] < topk_length[start:end, None]

        safe_indices = chunk_indices.clamp(0, SKV - 1)
        selected = torch.index_select(
            kv_flat, 0, safe_indices.reshape(-1)
        ).view(chunk_size, TOPK, DQK)
        q_fp32 = q[start:end].float()
        selected_fp32 = selected.float()

        logits = torch.bmm(q_fp32, selected_fp32.transpose(1, 2))
        logits.mul_(sm_scale)
        logits.masked_fill_(~valid[:, None, :], float("-inf"))
        chunk_max = logits.amax(dim=-1)
        valid_rows = torch.isfinite(chunk_max)
        safe_max = torch.where(valid_rows, chunk_max, 0.0)
        exp_logits = torch.exp(logits - safe_max.unsqueeze(-1))
        sum_exp = exp_logits.sum(dim=-1)
        accumulator = torch.bmm(
            exp_logits.to(torch.bfloat16).float(),
            selected_fp32[..., :d_v],
        )
        chunk_lse = chunk_max + torch.log(sum_exp)
        if attn_sink is None:
            factor = 1.0 / sum_exp
        else:
            sum_exp_new_lse = torch.exp(chunk_lse) + torch.exp(
                attn_sink[None, :]
            )
            factor = torch.exp(chunk_max) / sum_exp_new_lse
        chunk_output = accumulator * factor.unsqueeze(-1)
        chunk_output = torch.where(
            valid_rows.unsqueeze(-1), chunk_output, 0.0
        )
        chunk_lse = torch.where(valid_rows, chunk_lse, float("inf"))
        output[start:end].copy_(chunk_output.to(q.dtype))
        max_logits[start:end].copy_(chunk_max)
        lse[start:end].copy_(chunk_lse)

    return output, max_logits, lse
