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

import unittest
import numpy as np
import random as random_utils
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest
import torch as torch_original


def _generate_inputs_for_native_mha():
    batch_size = random(1, 20)
    seq_len = random(1, 20)
    dim_per_head = random(1, 10)
    num_heads = random(3, 10)
    embed_dim = dim_per_head * num_heads
    # num_heads = 6
    in_proj_dim = embed_dim
    out_proj_dim = random(20, 40)

    device = random_device()
    query = random_tensor(3, batch_size, seq_len, embed_dim).to(device)
    key = random_tensor(3, batch_size, seq_len, embed_dim).to(device)
    value = random_tensor(3, batch_size, seq_len, embed_dim).to(device)
    qkv_weight = random_tensor(2, embed_dim * 3, in_proj_dim).to(device)
    qkv_bias = random_tensor(1, embed_dim * 3).to(device)
    proj_weight = random_tensor(2, out_proj_dim, in_proj_dim).to(device)
    proj_bias = random_tensor(1, out_proj_dim).to(device)
    mask = random_tensor(2, batch_size, seq_len, low=0, high=1).to(device) > 0.5

    return (
        query,
        key,
        value,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask,
    )


def _generate_inputs_for_nn_module(
    embed_dim, kdim=None, vdim=None, batch_first=False, device=None, dtype=None
):
    batch_size = random(1, 20)
    tgt_len = random(1, 20)
    src_len = random(1, 20)
    is_batched = random_bool()
    kdim = embed_dim if kdim is None else kdim
    vdim = embed_dim if vdim is None else vdim

    if is_batched and batch_first:
        query_shape = (batch_size, tgt_len, embed_dim)
        key_shape = (batch_size, src_len, kdim)
        value_shape = (batch_size, src_len, vdim)
        key_padding_mask_shape = (batch_size, src_len)
    elif is_batched:
        query_shape = (tgt_len, batch_size, embed_dim)
        key_shape = (src_len, batch_size, kdim)
        value_shape = (src_len, batch_size, vdim)
        key_padding_mask_shape = (batch_size, src_len)
    else:
        query_shape = (tgt_len, embed_dim)
        key_shape = (src_len, kdim)
        value_shape = (src_len, vdim)
        key_padding_mask_shape = (src_len,)

    query = random_tensor(len(query_shape), *query_shape).to(device)
    key = random_tensor(len(key_shape), *key_shape).to(device)
    value = random_tensor(len(value_shape), *value_shape).to(device)
    key_padding_mask = random_tensor(len(key_padding_mask_shape), *key_padding_mask_shape).to(device) > 0
    attn_mask = random_tensor(2, tgt_len, src_len).to(device) > 0

    return query, key, value, key_padding_mask, attn_mask

def _align_params(flow_module: flow.nn.Module, torch_module):
    state_dict = flow_module.state_dict()
    for k, v in state_dict.items():
        with torch_original.no_grad():
            torch_module.get_parameter(k).copy_(torch_original.Tensor(v.numpy()).to(torch_module.get_parameter(k).device))
    return flow_module, torch_module

def _test_nn_module(test_case):
    dim_per_head = random_utils.randint(10, 20)
    num_heads = random_utils.randint(1, 10)
    embed_dim = dim_per_head * num_heads
    vdim = random_utils.choice([embed_dim, random_utils.randint(10, 200)])
    kdim = random_utils.choice([embed_dim, random_utils.randint(10, 200)])
    # dropout = random_utils.choice([0, 0, random_utils.random()])
    dropout = 0
    batch_first = random_bool().value()
    device = random_device().value()
    bias = random_bool()
    add_bias_kv = random_bool()
    add_zero_attn = random_bool()


    query, key, value, key_padding_mask, attn_mask = _generate_inputs_for_nn_module(
        embed_dim, kdim=kdim, vdim=vdim, device=device, batch_first=batch_first
    )

    torch_mha = torch_original.nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        batch_first=batch_first,
        kdim=kdim,
        vdim=vdim,
        device=device,
    )

    flow_mha = flow.nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        batch_first=batch_first,
        kdim=kdim,
        vdim=vdim,
        device=device,
    )

    flow_mha, torch_mha = _align_params(flow_mha, torch_mha)

    torch_results = torch_mha(
        query.pytorch, key.pytorch, value.pytorch, key_padding_mask.pytorch, need_weights=True, attn_mask=attn_mask.pytorch
    )

    flow_results = flow_mha(
        query.oneflow, key.oneflow, value.oneflow, key_padding_mask.oneflow, need_weights=True, attn_mask=attn_mask.oneflow
    )
    
    test_case.assertTrue(np.allclose(torch_results[0].detach().cpu().numpy(), flow_results[0].detach().numpy(), 1e-4, 1e-4))
    if torch_results[1] is not None and flow_results[1] is not None:
        test_case.assertTrue(np.allclose(torch_results[1].detach().cpu().numpy(), flow_results[1].detach().numpy(), 1e-4, 1e-4))

    torch_results[0].sum().backward()
    flow_results[0].sum().backward()

    for name, param in flow_mha.named_parameters():
        print(name)
        torch_grad = torch_mha.get_parameter(name).grad.detach().cpu().numpy()
        flow_grad = param.grad.detach().numpy()
        test_case.assertTrue(np.allclose(torch_grad, flow_grad, 1e-4, 1e-4))





def _test_multiheadattention(test_case):
    batch_size = 4
    seq_len = 13
    emb_dim = 35
    num_heads = 5
    in_proj_dim = emb_dim
    out_proj_dim = 32

    query = flow.randn(batch_size, seq_len, emb_dim).cuda()
    key = flow.randn(batch_size, seq_len, emb_dim).cuda()
    value = flow.randn(batch_size, seq_len, emb_dim).cuda()
    qkv_weight = flow.randn(emb_dim * 3, in_proj_dim).cuda()
    qkv_bias = flow.randn(emb_dim * 3).cuda()
    proj_weight = flow.randn(out_proj_dim, in_proj_dim).cuda()
    proj_bias = flow.randn(out_proj_dim,).cuda()
    mask = (flow.randn(batch_size, seq_len) > 0).cuda()

    query = flow.randn(batch_size, seq_len, emb_dim)
    key = flow.randn(batch_size, seq_len, emb_dim)
    value = flow.randn(batch_size, seq_len, emb_dim)
    qkv_weight = flow.randn(emb_dim * 3, in_proj_dim)
    qkv_bias = flow.randn(emb_dim * 3)
    proj_weight = flow.randn(out_proj_dim, in_proj_dim)
    proj_bias = flow.randn(out_proj_dim,)
    mask = flow.randn(batch_size, seq_len) > 0
    # mask = flow.zeros(batch_size, seq_len).bool().cuda()
    print(mask)

    flow_result = flow._native_multi_head_attention(
        query,
        key,
        value,
        emb_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=mask,
        need_weights=True,
        average_attn_weights=True,
        mask_type=1
        # mask=N, False, False, 1
    )
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias, mask = map(
        lambda x: torch_original.tensor(x.numpy()),
        [query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias, mask],
    )
    torch_result = torch_original._native_multi_head_attention(
        query,
        key,
        value,
        emb_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=mask,
        need_weights=True,
        average_attn_weights=True,
        mask_type=1,
    )
    print("flow_result: ", flow_result[0].mean())
    print("torch_result: ", torch_result[0].mean())


@flow.unittest.skip_unless_1n1d()
class TestMultiHeadAttentionModule(flow.unittest.TestCase):
    @autotest(n=10)
    def test_native_multi_head_attention(test_case):
        (
            query,
            key,
            value,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
        ) = _generate_inputs_for_native_mha()
        return torch._native_multi_head_attention(
            query,
            key,
            value,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask=mask,
            need_weights=True,
            average_attn_weights=True,
            mask_type=1,
        )

    @autotest(n=10)
    def test_nn_module(test_case):
        _test_nn_module(test_case)
        # dim_per_head = random_utils.randint(10, 20)
        # num_heads = random_utils.randint(1, 10)
        # embed_dim = dim_per_head * num_heads
        # vdim = random_utils.choice([embed_dim, random_utils.randint(10, 200)])
        # kdim = random_utils.choice([embed_dim, random_utils.randint(10, 200)])
        # dropout = random_utils.choice([0, 0, random_utils.random()])
        # batch_first = random_bool().value()
        # device = random_device()

        # query, key, value, key_padding_mask, attn_mask = _generate_inputs_for_nn_module(
        #     embed_dim, kdim=kdim, vdim=vdim, device=device, batch_first=batch_first
        # )

        # mha = torch.nn.MultiheadAttention(
        #     embed_dim,
        #     num_heads,
        #     dropout=dropout,
        #     bias=random_bool(),
        #     add_bias_kv=random_bool(),
        #     add_zero_attn=random_bool(),
        #     batch_first=batch_first,
        #     kdim=kdim,
        #     vdim=vdim,
        #     device=device,
        # )
        # return mha(
        #     query, key, value, key_padding_mask, need_weights=True, attn_mask=attn_mask
        # )

    # def test_multiheadattention(test_case):
    #     _test_multiheadattention(test_case)


if __name__ == "__main__":
    unittest.main()
