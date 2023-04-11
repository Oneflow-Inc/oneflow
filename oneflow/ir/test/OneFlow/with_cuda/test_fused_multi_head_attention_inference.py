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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: oneflow.softmax
# CHECK-NOT: oneflow.batch_matmul


import unittest
import numpy as np

import math
import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
os.environ["ONEFLOW_MLIR_CSE"] = "0"

import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig


def _ref(query, key, value, num_heads, causal=False):
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).permute(0, 2, 3, 1)
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )
    scores = flow.matmul(query, key) / math.sqrt(query.shape[-1])
    if causal:
        causal_mask = flow.triu(
            flow.ones(
                scores.shape[-2], scores.shape[-1], dtype=flow.bool, device="cuda"
            ),
            1,
        )
        scores = flow.masked_fill(scores, causal_mask, float("-inf"))
    attn = flow.softmax(scores, dim=-1)
    out = flow.matmul(attn, value)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out


def _ref2(query, key, value, num_heads, causal=False):
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).permute(0, 2, 1, 3)
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).permute(
        0, 2, 1, 3
    )
    query = query.reshape(-1, query.shape[2], query.shape[3])
    key = key.reshape(-1, key.shape[2], key.shape[3]).permute(0, 2, 1)
    value = value.reshape(-1, value.shape[2], value.shape[3])

    scale = 1 / math.sqrt(query.shape[-1])

    scores = flow.baddbmm(
        flow.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        ),
        query,
        key,
        beta=0,
        alpha=scale,
    )

    if causal:
        causal_mask = flow.triu(
            flow.ones(
                scores.shape[-2], scores.shape[-1], dtype=flow.bool, device="cuda"
            ),
            1,
        )
        scores = flow.masked_fill(scores, causal_mask, float("-inf"))
    attn = flow.softmax(scores, dim=-1)
    out = flow.matmul(attn, value)
    out = out.reshape(-1, num_heads, out.shape[1], out.shape[2])
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)

    return out


def _fused_mha(query, key, value, num_heads, causal=False):
    return flow._C.fused_multi_head_attention_inference(
        query, key, value, num_heads, causal=causal
    )


class GraphToRun(flow.nn.Graph):
    def __init__(self, ref=None, num_heads=None, causal=False):
        super().__init__()
        self.ref = ref
        self.causal = causal
        self.num_heads = num_heads

    def build(self, query, key, value):
        return self.ref(query, key, value, self.num_heads, self.causal)


def _test_fused_multi_head_attention_inference(
    test_case,
    batch_size,
    num_heads,
    query_seq_len,
    kv_seq_len,
    query_head_size,
    value_head_size,
    dtype,
    graph_builder,
    ref,
    causal=False,
):

    query = flow.randn(
        (batch_size, query_seq_len, num_heads * query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    key = flow.randn(
        (batch_size, kv_seq_len, num_heads * query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    value = flow.randn(
        (batch_size, kv_seq_len, num_heads * value_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)

    g = graph_builder(ref=ref, num_heads=num_heads, causal=causal)
    ref_out = ref(query, key, value, num_heads, causal).numpy()
    fused_out = _fused_mha(query, key, value, num_heads, causal).numpy()
    g_out = g(query, key, value).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))
    test_case.assertTrue(np.allclose(ref_out, g_out, atol=1e-2, rtol=1e-2))


@flow.unittest.skip_unless_1n1d()
@unittest.skipUnless(oneflow.sysconfig.with_cuda(), "needs -DBUILD_CUDA=ON")
class TestFusedMultiHeadAttentionInference(flow.unittest.TestCase):
    def test_multi_head_attention_inference(test_case):
        # test_case,batch_size, num_heads,query_seq_len, kv_seq_len,query_head_size,value_head_size,dtype
        for ref in [_ref, _ref2]:
            _test_fused_multi_head_attention_inference(
                test_case, 2, 8, 4096, 4096, 40, 40, flow.float16, GraphToRun, ref
            )
            _test_fused_multi_head_attention_inference(
                test_case, 2, 8, 4096, 77, 40, 40, flow.float16, GraphToRun, ref
            )
            _test_fused_multi_head_attention_inference(
                test_case, 2, 8, 1024, 1024, 80, 80, flow.float16, GraphToRun, ref
            )
            _test_fused_multi_head_attention_inference(
                test_case, 2, 8, 1024, 77, 80, 80, flow.float16, GraphToRun, ref
            )
            _test_fused_multi_head_attention_inference(
                test_case, 2, 8, 256, 256, 160, 160, flow.float16, GraphToRun, ref
            )
            _test_fused_multi_head_attention_inference(
                test_case, 2, 8, 256, 77, 160, 160, flow.float16, GraphToRun, ref
            )


if __name__ == "__main__":
    unittest.main()
