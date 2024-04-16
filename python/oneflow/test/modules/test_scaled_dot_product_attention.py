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
import math
import itertools
import os

import oneflow as flow


def _scaled_dot_product_attention(
    query, key, value,
):
    # input dims will equal 3 or 4.
    if key.ndim == 4:
        key = key.permute(0, 1, 3, 2)
    elif key.ndim == 3 :
        key = key.permute(0, 2, 1)
    scores = flow.matmul(query, key) / math.sqrt(query.shape[-1])
    attn = flow.softmax(scores, dim=-1)
    out = flow.matmul(attn, value)
    return out


def _test_scaled_dot_product_attention(
    test_case,
    batch_size,
    num_head_pair,
    seq_len_pair,
    head_size,
    dtype,
):
    num_heads = num_head_pair[0]
    num_heads_k = num_head_pair[1]
    seq_len_q = seq_len_pair[0]
    seq_len_kv = seq_len_pair[1]
    query = flow.randn(
        (batch_size, num_heads, seq_len_q, head_size), 
        device="cuda", 
        dtype=flow.float,
    ).to(dtype)
    key = flow.randn(
        (batch_size, num_heads_k, seq_len_kv, head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    value = flow.randn(
        (batch_size, num_heads_k, seq_len_kv, head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)

    fused_out = flow._C.scaled_dot_product_attention(
        query=query, key=key, value=value,
    ).cpu().numpy()
    if num_heads == num_heads_k :
        ref_out = _scaled_dot_product_attention(query, key, value,).cpu().numpy()
    else : # For GQA
        ref_out = flow.empty(query.shape, device='cuda', dtype=dtype)
        stride = num_heads / num_heads_k
        for i in range(0, num_heads) :
            j = int(i / stride)
            ref_out[:, i, :, :] = _scaled_dot_product_attention(query[:, i, :, :], key[:, j, :, :], value[:, j, :, :])

    if dtype == flow.float16 :
        test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))
    elif dtype == flow.bfloat16 :
        test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-1, rtol=1e-1))
    else :
        test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-3, rtol=1e-3))



@flow.unittest.skip_unless_1n1d()
class TestScaledDotProductAttention(flow.unittest.TestCase):
    def test_scaled_dot_product_attention(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_scaled_dot_product_attention]
        args_dict["batchsize"] = [1, 2, 4]
        args_dict["num_head_pair"] = [[16, 16], [16, 8]]
        args_dict["seqlen_pair"] = [[4096, 4096], [4096, 77], [1024, 1024], [1024, 77]]
        args_dict["head_size"] = [40, 80, 160, 41]
        args_dict["dtype"] = [flow.float16, flow.bfloat16]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])

if __name__ == "__main__":
    unittest.main()
