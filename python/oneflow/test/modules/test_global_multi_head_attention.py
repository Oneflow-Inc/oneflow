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
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
import numpy as np
import torch


def _test_multi_head_attention_with_random_data(
    test_case, placement, input_sbp, m, name, embed_dim, query
):
    query = (
        m.from_numpy(query).to(m.float32).to_global(placement=placement, sbp=input_sbp)
        if name == "flow"
        else m.from_numpy(query).to(m.float32)
    )
    key = query
    value = query
    num_heads = 4

    bias_sbp = random_sbp(placement, max_dim=1, except_partial_sum=True).value()

    in_proj_weight = m.nn.Parameter(
        m.ones((3 * embed_dim, embed_dim)).to_global(placement=placement, sbp=input_sbp)
        if name == "flow"
        else m.ones((3 * embed_dim, embed_dim))
    )
    in_proj_bias = m.nn.Parameter(
        m.ones((3 * embed_dim)).to_global(placement=placement, sbp=bias_sbp)
        if name == "flow"
        else m.ones((3 * embed_dim))
    )
    bias_k = m.nn.Parameter(
        m.ones((1, 1, embed_dim)).to_global(placement=placement, sbp=input_sbp)
        if name == "flow"
        else m.ones((1, 1, embed_dim))
    )
    bias_v = m.nn.Parameter(
        m.ones((1, 1, embed_dim)).to_global(placement=placement, sbp=input_sbp)
        if name == "flow"
        else m.ones((1, 1, embed_dim))
    )
    add_zero_attn = True
    dropout_p = 0  # must be 0
    out_proj_weight = m.nn.Parameter(
        m.ones((embed_dim, embed_dim)).to_global(placement=placement, sbp=input_sbp)
        if name == "flow"
        else m.ones((embed_dim, embed_dim))
    )
    out_proj_bias = m.nn.Parameter(
        m.ones((embed_dim)).to_global(placement=placement, sbp=bias_sbp)
        if name == "flow"
        else m.ones((embed_dim))
    )

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


class TestMultiHeadAttentionModule(flow.unittest.TestCase):
    @globaltest
    def test_multi_head_attention_with_random_data(test_case):
        embed_dim = 16
        for placement in all_placement():
            for input_sbp in all_sbp(placement, max_dim=2):
                query = np.random.rand(4, 8, embed_dim)
                y1_flow, y2_flow = _test_multi_head_attention_with_random_data(
                    test_case,
                    placement,
                    input_sbp,
                    flow,
                    "flow",
                    embed_dim,
                    query.copy(),
                )
                y1_torch, y2_torch = _test_multi_head_attention_with_random_data(
                    test_case,
                    placement,
                    input_sbp,
                    torch,
                    "torch",
                    embed_dim,
                    query.copy(),
                )

                test_case.assertTrue(
                    np.allclose(
                        y1_torch.detach().cpu().numpy(),
                        y1_flow.detach().cpu().numpy(),
                        rtol=1e-4,
                        atol=1e-3,
                    )
                )
                if y2_torch is None:
                    test_case.assertTrue(y2_flow is None)
                else:
                    test_case.assertTrue(
                        np.allclose(
                            y2_torch.detach().cpu().numpy(),
                            y2_flow.detach().cpu().numpy(),
                            rtol=1e-4,
                            atol=1e-3,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
