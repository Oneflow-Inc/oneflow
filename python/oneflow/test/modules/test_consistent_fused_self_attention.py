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


def _test_fused_self_attention(
    test_case, batch_size, seq_len, num_heads, head_size, placement, sbp
):
    hidden_size = num_heads * 3 * head_size
    x = random_tensor(3, seq_len, batch_size, hidden_size).oneflow

    fused_input = x.to_global(placement=placement, sbp=sbp)
    fused_input.retain_grad()
    (fused_qmk, fused_v) = flow._C.fused_self_attention(
        fused_input, head_size=head_size, alpha=1.0,
    )
    fused_atten = flow.matmul(fused_qmk, fused_v)
    fused_atten.sum().backward()

    origin_input = x.to_local().to(placement.type)
    origin_input.retain_grad()
    reshape_input = flow.reshape(origin_input, (seq_len, batch_size, -1, 3 * head_size))

    origin_q = flow.slice(
        reshape_input,
        slice_tup_list=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
            [0, head_size, 1],
        ],
    ).permute(1, 2, 0, 3)
    origin_k = flow.slice(
        reshape_input,
        slice_tup_list=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
            [head_size, 2 * head_size, 1],
        ],
    ).permute(1, 2, 0, 3)
    origin_v = flow.slice(
        reshape_input,
        slice_tup_list=[
            [None, None, None],
            [None, None, None],
            [None, None, None],
            [2 * head_size, 3 * head_size, 1],
        ],
    ).permute(1, 2, 0, 3)

    origin_k = origin_k.transpose(2, 3)
    origin_qmk = flow.matmul(origin_q, origin_k)
    origin_atten = flow.matmul(origin_qmk, origin_v)
    origin_atten.sum().backward()

    test_case.assertTrue(
        np.allclose(fused_atten.numpy(), origin_atten.numpy(), atol=1e-4, rtol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(
            fused_input.grad.numpy(), origin_input.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )


class TestFusedSelfAttention(flow.unittest.TestCase):
    @globaltest
    def test_fused_self_attention(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [8]
        arg_dict["seq_len"] = [8, 16]
        arg_dict["num_heads"] = [8]
        arg_dict["head_size"] = [8, 16]

        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                if placement.type != "cuda":
                    continue
                for sbp in all_sbp(placement, max_dim=2):
                    _test_fused_self_attention(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()
