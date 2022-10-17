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
from packaging import version


def _generate_inputs_for_native_mha():
    batch_size = random(1, 20)
    seq_len = random(1, 20)
    dim_per_head = random(1, 10)
    num_heads = random(3, 10)
    embed_dim = dim_per_head * num_heads
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
    mask = random_tensor(2, batch_size, seq_len, low=0, high=1).to(device) > 0

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
    key_padding_mask = (
        random_tensor(len(key_padding_mask_shape), *key_padding_mask_shape).to(device)
        > 0
    )
    attn_mask = random_tensor(2, tgt_len, src_len).to(device) > 0

    return query, key, value, key_padding_mask, attn_mask


def _align_params(flow_module: flow.nn.Module, torch_module):
    state_dict = flow_module.state_dict()
    for k, v in state_dict.items():
        with torch_original.no_grad():
            torch_module.get_parameter(k).copy_(
                torch_original.Tensor(v.numpy()).to(
                    torch_module.get_parameter(k).device
                )
            )
    return flow_module, torch_module


def _test_mha_nn_module(test_case):
    dim_per_head = random_utils.randint(10, 20)
    num_heads = random_utils.randint(1, 10)
    embed_dim = dim_per_head * num_heads
    vdim = random_utils.choice([embed_dim, random_utils.randint(10, 200)])
    kdim = random_utils.choice([embed_dim, random_utils.randint(10, 200)])
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
        query.pytorch,
        key.pytorch,
        value.pytorch,
        key_padding_mask.pytorch,
        need_weights=True,
        attn_mask=attn_mask.pytorch,
    )

    flow_results = flow_mha(
        query.oneflow,
        key.oneflow,
        value.oneflow,
        key_padding_mask.oneflow,
        need_weights=True,
        attn_mask=attn_mask.oneflow,
    )

    test_case.assertTrue(
        np.allclose(
            torch_results[0].detach().cpu().numpy(),
            flow_results[0].detach().numpy(),
            1e-4,
            1e-4,
        )
    )
    if torch_results[1] is not None and flow_results[1] is not None:
        test_case.assertTrue(
            np.allclose(
                torch_results[1].detach().cpu().numpy(),
                flow_results[1].detach().numpy(),
                1e-4,
                1e-4,
            )
        )

    torch_results[0].sum().backward()
    flow_results[0].sum().backward()

    for name, param in flow_mha.named_parameters():
        print(name)
        torch_grad = torch_mha.get_parameter(name).grad.detach().cpu().numpy()
        flow_grad = param.grad.detach().numpy()
        test_case.assertTrue(np.allclose(torch_grad, flow_grad, 1e-4, 1e-4))


@flow.unittest.skip_unless_1n1d()
class TestMultiHeadAttentionModule(flow.unittest.TestCase):
    @unittest.skipIf(version.parse(torch_original.__version__) < version.parse("1.14.0"), "torch below 1.14.0 has not torch._native_multi_head_attention")
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
    def test_mha_nn_module(test_case):
        _test_mha_nn_module(test_case)

    @autotest(n=10)
    def test_mha_functional(test_case):
        device = random_device()

        is_batched = random_bool().value()
        use_separate_proj_weight = random_bool().value()
        bias_k_v = random_bool().value()
        need_weights = random_bool().value()

        batch_size = random(1, 20)
        tgt_len = random(1, 20)
        src_len = random(1, 20)
        num_heads = random(3, 10)
        dims_per_head = random(10, 20)
        embed_dim = num_heads * dims_per_head

        if use_separate_proj_weight:
            kdim = random(30, 200)
            vdim = random(30, 200)
        else:
            kdim = embed_dim
            vdim = embed_dim

        if is_batched:
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
        key_padding_mask = (
            random_tensor(len(key_padding_mask_shape), *key_padding_mask_shape).to(
                device
            )
            > 0
        )
        attn_mask = random_tensor(2, tgt_len, src_len).to(device) > 0
        out_proj_weight = random_tensor(2, embed_dim, embed_dim).to(device)
        out_proj_bias = random_tensor(1, embed_dim).to(device)

        if bias_k_v:
            bias_k = random_tensor(3, 1, 1, embed_dim).to(device)
            bias_v = random_tensor(3, 1, 1, embed_dim).to(device)
            static_k, static_v = None, None
        elif is_batched:
            static_k = random_tensor(
                3, batch_size * num_heads, src_len, dims_per_head
            ).to(device)
            static_v = random_tensor(
                3, batch_size * num_heads, src_len, dims_per_head
            ).to(device)
            bias_k, bias_v = None, None
        else:
            static_k = random_tensor(3, num_heads, src_len, dims_per_head).to(device)
            static_v = random_tensor(3, num_heads, src_len, dims_per_head).to(device)
            bias_k, bias_v = None, None

        if use_separate_proj_weight:
            q_proj_weight = random_tensor(2, embed_dim, embed_dim).to(device)
            k_proj_weight = random_tensor(2, embed_dim, kdim).to(device)
            v_proj_weight = random_tensor(2, embed_dim, vdim).to(device)
            in_proj_weight = None
            in_proj_bias = None
        else:
            q_proj_weight, k_proj_weight, v_proj_weight = None, None, None
            in_proj_weight = random_tensor(2, embed_dim * 3, embed_dim).to(device)
            in_proj_bias = random_tensor(1, embed_dim * 3,).to(device)

        result = torch.nn.functional.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=embed_dim,
            num_heads=num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            bias_k=bias_k,
            bias_v=bias_v,
            add_zero_attn=random_bool(),
            dropout_p=0,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            training=False,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=random_bool(),
        )

        if need_weights:
            return result
        else:
            return result[0]


if __name__ == "__main__":
    unittest.main()
