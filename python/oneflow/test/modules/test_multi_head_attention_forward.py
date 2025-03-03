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
from itertools import product


def _multi_head_attention(m, query, device, add_zero_attn, embed_dim, num_heads):
    device = m.device(device)
    query = m.from_numpy(query).to(device).to(m.float32)
    key = query
    value = query

    in_proj_weight = m.nn.Parameter(m.ones((3 * embed_dim, embed_dim))).to(
        device
    )
    in_proj_bias = m.nn.Parameter(m.ones(3 * embed_dim)).to(device)
    bias_k = m.nn.Parameter(m.ones((1, 1, embed_dim))).to(device)
    bias_v = m.nn.Parameter(m.ones((1, 1, embed_dim))).to(device)
    dropout_p = 0 # must be 0
    out_proj_weight = m.nn.Parameter(m.ones((embed_dim, embed_dim))).to(
        device
    )
    out_proj_bias = m.nn.Parameter(m.ones((embed_dim))).to(device)

    y1, y2 = m.nn.functional.multi_head_attention_forward(
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
    return y1, y2


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_multi_head_attention_with_torch(test_case):
        device_list = ["cpu", "cuda"]
        add_zero_attn_list = [True, False]

        for device, aza  in product(device_list, add_zero_attn_list):
            param = {
                "device": device,
                "add_zero_attn": aza,
                "embed_dim": 16,
                "num_heads": 4
            }
            query = np.random.rand(4, 8, param["embed_dim"])
            y1_torch, y2_torch = _multi_head_attention(torch, query.copy(), **param)
            y1_flow, y2_flow = _multi_head_attention(flow, query.copy(), **param)

            test_case.assertTrue(
                np.allclose(
                    y1_torch.detach().cpu().numpy(), y1_flow.detach().cpu().numpy(), rtol=1e-4, atol=1e-3
                )
            )
            if y2_torch is None:
                test_case.assertTrue(y2_flow is None)
            else:
                test_case.assertTrue(
                    np.allclose(
                        y2_torch.detach().cpu().numpy(), y2_flow.detach().cpu().numpy(), rtol=1e-4, atol=1e-3
                    )
                )


if __name__ == "__main__":
    unittest.main()
    
