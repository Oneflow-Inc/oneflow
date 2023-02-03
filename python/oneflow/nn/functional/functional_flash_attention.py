"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from typing import List, Optional

import oneflow as flow
from oneflow.framework.tensor import Tensor
from einops import rearrange, repeat

# from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexFirstAxis(flow.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return flow.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        first_axis_dim = ctx.first_axis_dim
        grad_input = flow.zeros(
            [first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


def _unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (total_batch, seqlen_q, seqlen_k)
        attention_mask: 
            (batch, 1, 1, seqlen_k), bool / int, 1 means valid and 0 means not valid.

    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    if len(attention_mask.shape) == 4:
        assert attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1
        attention_mask = attention_mask.squeeze(1).squeeze(1)
    seqlens_in_batch = attention_mask.sum(dim=-1)
    bias_nonzere_indices = (
        flow.nonzero(attention_mask, as_tuple=False)[..., -1].flatten().to(flow.int32)
    )
    indices = flow.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = flow.nn.functional.pad(
        flow.cumsum(seqlens_in_batch, dim=0, dtype=flow.int32), (1, 0)
    )
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.

    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        bias_nonzere_indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    mask: Tensor = None,
    bias: Tensor = None,
    unpad_kv=False,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> Tensor:
    r""" flash attention
    Adapted from https://github.com/HazyResearch/flash-attention
        
    Arguments:
        query: flow.Tensor with shape [*, batch_size, num_heads, seq_length_q, head_dim].
        key: flow.Tensor with shape [*, batch_size, num_heads, seq_length_k, head_dim].
        value: flow.Tensor with shape [*, batch_size, num_heads, seq_length_k, head_dim].

        mask: Optional, default None. 
            If provided, flow.Tensor with shape [batch_size, s1, s2, seq_length_k], where (s1=1 or s1=num_heads)
            and (s2=1 or s2=seq_length_q).
        bias: Optional, default None.
            If provided, flow.Tensor with shape [s0, num_heads, seq_length_q, seq_length_k],
            where (s0=1 or s0=batch_size).
        unpad_kv: whether unpad key and value, could be True, False, or 'auto'.
            `mask` should be provided if you set unpad_kv to 'auto'.
        causal:  If causal=True,  default False. 
        dropout_p: dropout rate, perform dropout operation when dropout_p > 0.
    
    Returns:
        flow.Tensor with shape [*, batch_size, num_heads, seq_length_q, head_dim].
        
    This function provides a faster way **with the loss of numerical precision** to compute:
        `flow.matmul(flow.softmax(flow.matmul(q, k.transpose(-1,-2))/flow.sqrt(head_dim) + mask + bias, dim=-1), value)`
    """

    batch_dims = query.shape[:-3]
    batch_size, no_heads, seqlen_q, c = query.shape[-4:]
    seqlen_k = key.shape[-2]
    dtype = query.dtype

    # convert to half
    if query.dtype not in [flow.float16, flow.bfloat16]:
        query = query.half()
        key = key.half()
        value = value.half()
    if bias is not None:
        bias = bias.to(dtype)

    if unpad_kv == "auto":
        assert mask is not None, "mask shoule be provided if set unpad_kv to 'auto'"
        nonzero_sum = mask.sum()
        total_n = mask.shape.numel()
        unpad_kv = nonzero_sum < total_n * 0.2

    # flow.gather do not support bf16 yet
    if dtype == flow.bfloat16:
        unpad_kv = False

    # [*, B, N, H, C]
    query = query.transpose(-2, -3)
    key = key.transpose(-2, -3)
    value = value.transpose(-2, -3)

    # [B_flat, N, H, C]
    query = query.reshape(-1, *query.shape[-3:])
    key = key.reshape(-1, *key.shape[-3:])
    value = value.reshape(-1, *value.shape[-3:])

    if unpad_kv == False:
        if mask is not None and (
            mask.shape[-2] != 1 or mask.shape[-3] == 1
        ):  # or bias.shape[-4] != 1:
            if mask.dtype in [flow.int32, flow.int64, flow.bool]:
                mask = ((1 - mask) * -10000.0).to(query.dtype)
            cu_seqlens_q = (
                flow.tensor(list(range(0, (batch_size + 1) * seqlen_q, seqlen_q)))
                .cuda()
                .to(flow.int32)
            )
            cu_seqlens_k = (
                flow.tensor(list(range(0, (batch_size + 1) * seqlen_k, seqlen_k)))
                .cuda()
                .to(flow.int32)
            )

            out, lse = flow._C.flash_attention(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                mask=mask,
                bias=bias,
                softmax_scale=1 / (c ** 0.5),
                causal=causal,
                dropout_rate=0,
            )
            out = out.to(dtype)
            out = out.reshape(*batch_dims, seqlen_q, no_heads, c).transpose(-2, -3)
            return out

    if mask is not None and mask.dtype != flow.int32:
        if mask.dtype in [flow.float16, flow.bfloat16, flow.float32, flow.float64]:
            mask = mask > -1.0
        mask = mask.to(flow.int32)
    # Flattened batch size
    batch_size = query.shape[0]
    max_seqlen_q = seqlen_q
    # [B_flat * N, H, C]
    query = query.reshape(-1, *query.shape[-2:])

    q_cu_seqlens = flow.arange(
        0,
        (batch_size + 1) * seqlen_q,
        step=seqlen_q,
        dtype=flow.int32,
        device=query.device,
    )

    # [B_flat, N, 2, H, C]
    kv = flow.stack([key, value], dim=-3)
    kv_shape = kv.shape
    # [B_flat, N, 2 * H * C]
    kv = kv.reshape(*kv.shape[:-3], -1)

    kv_unpad, bias_nonzero_indices, kv_cu_seqlens, kv_max_s = _unpad_input(kv, mask)
    kv_cu_seqlens = kv_cu_seqlens.to(flow.int32)
    kv_max_s = int(kv_max_s)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])  # [nonzeros, 2, H, C]

    # we need pass the origin seqlens `seqlen_k`, for convenient we store it at
    # the begining of `bias_nonzero_indices`, this will be used by
    # gmem_tile.h/Gmem_tile_mma_bias & Gmem_tile_mma_ds's ptr_, so be careful here!
    bias_nonzero_indices = flow.concat(
        [
            flow.tensor(
                [seqlen_k], dtype=flow.int32, device=bias_nonzero_indices.device
            ),
            bias_nonzero_indices,
        ]
    )
    out, softmax_lse = flow._C.flash_attention(
        query,  # total_q * num_heads * head_size
        kv_unpad[..., 0, :, :],  # total_k * num_heads * head_size
        kv_unpad[..., 1, :, :],  # total_k * num_heads * head_size
        q_cu_seqlens,
        kv_cu_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=kv_max_s,
        indices=bias_nonzero_indices,
        mask=None,
        bias=bias,
        softmax_scale=1 / (c ** 0.5),
        causal=causal,
        dropout_rate=dropout_p,
    )

    out = out.to(dtype)
    out = out.reshape(*batch_dims, seqlen_q, no_heads, c).transpose(-2, -3)
    return out
