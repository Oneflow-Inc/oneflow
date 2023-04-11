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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
import torch


def _multi_head_attention_torch(test_case, device, query):
    device = torch.device(device)
    embed_dim = 16
    num_heads = 4
    query = torch.from_numpy(query).to(device).to(torch.float32)
    key = query
    value = query

    in_proj_weight = torch.nn.Parameter(torch.ones((3 * embed_dim, embed_dim))).to(
        device
    )
    in_proj_bias = torch.nn.Parameter(torch.ones(3 * embed_dim)).to(device)
    bias_k = torch.nn.Parameter(torch.ones((1, 1, embed_dim))).to(device)
    bias_v = torch.nn.Parameter(torch.ones((1, 1, embed_dim))).to(device)
    add_zero_attn = True#random().to(bool)
    dropout_p = 0
    out_proj_weight = torch.nn.Parameter(torch.ones((embed_dim, embed_dim))).to(
        device
    )
    out_proj_bias = torch.nn.Parameter(torch.ones((embed_dim))).to(device)

    y, _ = torch.nn.functional.multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
    )
    return y


def _multi_head_attention_flow(test_case, device, query):
    device = flow.device(device)
    embed_dim = 16
    num_heads = 4
    query = flow.from_numpy(query).to(device).to(flow.float32)
    key = query
    value = query

    in_proj_weight = flow.nn.Parameter(flow.ones((3 * embed_dim, embed_dim))).to(
        device
    )
    in_proj_bias = flow.nn.Parameter(flow.ones(3 * embed_dim)).to(device)
    bias_k = flow.nn.Parameter(flow.ones((1, 1, embed_dim))).to(device)
    bias_v = flow.nn.Parameter(flow.ones((embed_dim))).to(device)
    add_zero_attn = True#random().to(bool)
    dropout_p = 0
    out_proj_weight = flow.nn.Parameter(flow.ones((embed_dim, embed_dim))).to(
        device
    )
    out_proj_bias = flow.nn.Parameter(flow.ones((embed_dim))).to(device)

    y, _ = flow.nn.functional.multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
    )
    return y


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_multi_head_attention_with_torch(test_case):
        device_list = ["cpu", "cuda"]
        for device in device_list:
            query = np.random.rand(4, 8, 16)
            y_torch = _multi_head_attention_torch(test_case, device, query.copy())
            y_flow = _multi_head_attention_flow(test_case, device, query.copy())
            test_case.assertTrue(
                np.allclose(
                    y_torch.detach().cpu().numpy(), y_flow.detach().cpu().numpy(), rtol=1e-4, atol=1e-3
                )
            )


if __name__ == "__main__":
    unittest.main()
    
