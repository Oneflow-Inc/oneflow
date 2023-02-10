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

import oneflow as flow


def _ref(query, key, value, num_heads, causal=False, attn_bias=None):
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
    if attn_bias is not None:
        scores = scores + attn_bias
    attn = flow.softmax(scores, dim=-1)
    out = flow.matmul(attn, value)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out


def _fused_mha(query, key, value, num_heads, causal=False, attn_bias=None):
    if attn_bias is not None and attn_bias.shape[-1] % 8 != 0:
        pad = 8 - attn_bias.shape[-1] % 8
        attn_bias = flow.pad(attn_bias, (0, pad), "constant", 0)
    return flow._C.fused_multi_head_attention_inference(
        query, key, value, num_heads, causal=causal, attn_bias=attn_bias
    )


def _test_fused_multi_head_attention_inference(
    test_case,
    batch_size,
    num_heads,
    query_seq_len,
    kv_seq_len,
    query_head_size,
    value_head_size,
    dtype,
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

    fused_out = _fused_mha(query, key, value, num_heads, causal).numpy()
    ref_out = _ref(query, key, value, num_heads, causal).numpy()

    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))


def _test_fused_multi_head_attention_inference_with_attn_bias(
    test_case,
    batch_size,
    num_heads,
    query_seq_len,
    kv_seq_len,
    query_head_size,
    value_head_size,
    dtype,
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

    attn_bias = flow.randn((kv_seq_len,), device="cuda", dtype=flow.float).to(dtype)
    ref_out = _ref(query, key, value, num_heads, causal, attn_bias).numpy()
    fused_out = _fused_mha(query, key, value, num_heads, causal, attn_bias).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (query_seq_len, kv_seq_len), device="cuda", dtype=flow.float
    ).to(dtype)
    ref_out = _ref(query, key, value, num_heads, causal, attn_bias).numpy()
    fused_out = _fused_mha(query, key, value, num_heads, causal, attn_bias).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (num_heads, query_seq_len, kv_seq_len), device="cuda", dtype=flow.float
    ).to(dtype)
    ref_out = _ref(query, key, value, num_heads, causal, attn_bias).numpy()
    fused_out = _fused_mha(query, key, value, num_heads, causal, attn_bias).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (batch_size, num_heads, query_seq_len, kv_seq_len),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    ref_out = _ref(query, key, value, num_heads, causal, attn_bias).numpy()
    fused_out = _fused_mha(query, key, value, num_heads, causal, attn_bias).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (num_heads, 1, kv_seq_len), device="cuda", dtype=flow.float
    ).to(dtype)
    ref_out = _ref(query, key, value, num_heads, causal, attn_bias).numpy()
    fused_out = _fused_mha(query, key, value, num_heads, causal, attn_bias).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))


@unittest.skipIf(True, "skip test")
@flow.unittest.skip_unless_1n1d()
class TestFusedMultiHeadAttentionInference(flow.unittest.TestCase):
    def test_multi_head_attention_inference(test_case):
        # test_case,batch_size, num_heads,query_seq_len, kv_seq_len,query_head_size,value_head_size,dtype
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 77, 40, 40, flow.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 1024, 80, 80, flow.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 77, 80, 80, flow.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 256, 160, 160, flow.float16
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 77, 160, 160, flow.float16
        )

        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 4096, 77, 40, 40, flow.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 1024, 80, 80, flow.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 1024, 77, 80, 80, flow.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 256, 160, 160, flow.float
        )
        _test_fused_multi_head_attention_inference(
            test_case, 2, 8, 256, 77, 160, 160, flow.float
        )

    def test_multi_head_attention_inference_with_attn_bias(test_case):
        # test_case,batch_size, num_heads,query_seq_len, kv_seq_len,query_head_size,value_head_size,dtype
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float16
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float16, True
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float, True
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float16
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float16, True
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float, True
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 77, 40, 40, flow.float, True
        )


if __name__ == "__main__":
    unittest.main()
