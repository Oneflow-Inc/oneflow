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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

import torch as torch_original


def _test_multiheadattention(test_case):
    batch_size = 4
    seq_len = 13
    emb_dim = 35
    num_heads = 5
    in_proj_dim = emb_dim
    out_proj_dim = 32

    query = flow.randn(batch_size, seq_len, emb_dim)
    key = flow.randn(batch_size, seq_len, emb_dim)
    value = flow.randn(batch_size, seq_len, emb_dim)
    qkv_weight = flow.randn(emb_dim * 3, in_proj_dim)
    qkv_bias = flow.randn(emb_dim * 3)
    proj_weight = flow.randn(out_proj_dim, in_proj_dim)
    proj_bias = flow.randn(out_proj_dim, )
    # mask = flow.randn(batch_size, seq_len) > 0
    mask = flow.zeros(batch_size, seq_len).bool()
    

    flow_result = flow.nn.functional.multi_head_attention_forward(
        query, key, value, emb_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
        mask=None, need_weights=True, average_attn_weights=True, mask_type=1
        # mask=N, False, False, 1
    )
    query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias, mask = map(
        lambda x: torch_original.tensor(x.numpy()), 
        [query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias, mask]
    )
    torch_result = torch_original._native_multi_head_attention(
        query, key, value, emb_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
        mask=None, need_weights=True, average_attn_weights=True, mask_type=1
    )
    print("flow_result: ", flow_result[1].mean())
    print("torch_result: ", torch_result[1].mean())

@flow.unittest.skip_unless_1n1d()
class TestMultiHeadAttentionModule(flow.unittest.TestCase):
    def test_multiheadattention(test_case):
        _test_multiheadattention(test_case)




if __name__ == "__main__":
    unittest.main()
