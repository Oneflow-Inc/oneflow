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


def _ref(
    query,
    key,
    value,
    num_heads,
    attn_mask_type="none",
    attn_bias=None,
    causal_diagonal_offset=0,
    query_seq_len=None,
    key_seq_len=None,
):
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 3, 1)
    value = value.permute(0, 2, 1, 3)
    scores = flow.matmul(query, key) / math.sqrt(query.shape[-1])
    if attn_mask_type == "causal_from_bottom_right":
        causal_diagonal_offset += key.shape[-1] - query.shape[-2]
    if (
        attn_mask_type == "causal_from_top_left"
        or attn_mask_type == "causal_from_bottom_right"
    ):
        causal_mask = flow.triu(
            flow.ones(
                scores.shape[-2], scores.shape[-1], dtype=flow.bool, device="cuda"
            ),
            causal_diagonal_offset + 1,
        )
        scores = flow.masked_fill(scores, causal_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    if query_seq_len is not None:
        scores = flow.masked_fill(
            scores,
            flow.arange(scores.shape[-2], device=query_seq_len.device).view(
                1, 1, scores.shape[-2], 1
            )
            >= query_seq_len.view(scores.shape[0], 1, 1, 1),
            float("-inf"),
        )
    if key_seq_len is not None:
        scores = flow.masked_fill(
            scores,
            flow.arange(scores.shape[-1], device=key_seq_len.device).view(
                1, 1, 1, scores.shape[-1]
            )
            >= key_seq_len.view(scores.shape[0], 1, 1, 1),
            float("-inf"),
        )
    attn = flow.softmax(scores, dim=-1)
    out = flow.matmul(attn, value)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out


def _to_layout(ts, layout, tensor_index, seq_len=None):
    if layout == "BMHK":
        return ts[tensor_index]
    elif layout == "BM(HK)":
        return ts[tensor_index].view(
            ts[tensor_index].shape[0], ts[tensor_index].shape[1], -1
        )
    elif layout == "MB(HK)":
        return (
            ts[tensor_index]
            .view(ts[tensor_index].shape[0], ts[tensor_index].shape[1], -1)
            .transpose(0, 1)
        )
    elif layout == "BHMK":
        return ts[tensor_index].transpose(1, 2)
    elif layout == "MBHK":
        return ts[tensor_index].transpose(0, 1)
    elif layout == "BM(H3K)":
        return flow.stack(ts, -2).view(ts[0].shape[0], ts[0].shape[1], -1)
    elif layout == "MB(H3K)":
        return (
            flow.stack(ts, -2).view(ts[0].shape[0], ts[0].shape[1], -1).transpose(0, 1)
        )
    elif layout == "BM(H2K)":
        return flow.stack(ts[1:], -2).view(ts[1].shape[0], ts[1].shape[1], -1)
    elif layout == "MB(H2K)":
        return (
            flow.stack(ts[1:], -2)
            .view(ts[1].shape[0], ts[1].shape[1], -1)
            .transpose(0, 1)
        )
    elif layout == "(BM)HK":
        t = ts[tensor_index]
        if seq_len is None:
            return t.view(-1, t.shape[-2], t.shape[-1])
        mask = flow.arange(t.shape[1], device=t.device).view(
            1, t.shape[1]
        ) < seq_len.view(t.shape[0], 1)
        return flow.masked_select(
            t, mask.view(mask.shape[0], mask.shape[1], 1, 1)
        ).view(-1, t.shape[-2], t.shape[-1])
    elif layout == "(BM)(HK)":
        t = ts[tensor_index]
        if seq_len is None:
            return t.view(-1, t.shape[-2] * t.shape[-1])
        mask = flow.arange(t.shape[1], device=t.device).view(
            1, t.shape[1]
        ) < seq_len.view(t.shape[0], 1)
        return flow.masked_select(
            t, mask.view(mask.shape[0], mask.shape[1], 1, 1)
        ).view(-1, t.shape[-2] * t.shape[-1])
    elif layout == "(BM)(H2K)":
        t = flow.stack(ts[1:], -2)
        if seq_len is None:
            return t.view(t.shape[0] * t.shape[1], -1)
        mask = flow.arange(t.shape[1], device=t.device).view(
            1, t.shape[1]
        ) < seq_len.view(t.shape[0], 1)
        return flow.masked_select(
            t, mask.view(mask.shape[0], mask.shape[1], 1, 1, 1)
        ).view(-1, t.shape[-3] * t.shape[-2] * t.shape[-1])
    elif layout == "(BM)(H3K)":
        t = flow.stack(ts, -2)
        if seq_len is None:
            return t.view(t.shape[0] * t.shape[1], -1)
        mask = flow.arange(t.shape[1], device=t.device).view(
            1, t.shape[1]
        ) < seq_len.view(t.shape[0], 1)
        return flow.masked_select(
            t, mask.view(mask.shape[0], mask.shape[1], 1, 1, 1)
        ).view(-1, t.shape[-3] * t.shape[-2] * t.shape[-1])
    else:
        raise NotImplementedError


def _fused_mha(
    query,
    key,
    value,
    num_heads,
    attn_mask_type="none",
    attn_bias=None,
    causal_diagonal_offset=0,
    query_layout="BM(HK)",
    key_layout="BM(HK)",
    value_layout="BM(HK)",
    output_layout="MB(HK)",
    query_seq_len=None,
    key_seq_len=None,
    use_kv_seq_len=False,
):
    batch_size = query.shape[0]
    query_max_seq_len = query.shape[1]
    query_head_size = query.shape[-1]
    key_max_seq_len = key.shape[1]
    ts = [query, key, value]
    query = _to_layout(ts, query_layout, 0, query_seq_len)
    if use_kv_seq_len:
        key = _to_layout(ts, key_layout, 1)
        value = _to_layout(ts, value_layout, 2)
    else:
        key = _to_layout(ts, key_layout, 1, key_seq_len)
        value = _to_layout(ts, value_layout, 2, key_seq_len)
    if query_seq_len is not None:
        query_seq_start = (
            flow.cumsum(flow.pad(query_seq_len, (1, 0)), dim=-1)
            .to(flow.int32)
            .to(query.device)
        )
    else:
        query_seq_start = None
        query_max_seq_len = None
    if key_seq_len is not None:
        if use_kv_seq_len:
            key_seq_start = flow.arange(
                0,
                key_max_seq_len * (batch_size + 1),
                key_max_seq_len,
                dtype=flow.int32,
                device=key_seq_len.device,
            )
        else:
            key_seq_start = (
                flow.cumsum(flow.pad(key_seq_len, (1, 0)), dim=-1)
                .to(flow.int32)
                .to(query.device)
            )
    else:
        key_seq_start = None
        key_max_seq_len = None
    if attn_bias is not None and attn_bias.shape[-1] % 8 != 0:
        pad = 8 - attn_bias.shape[-1] % 8
        attn_bias = flow.pad(attn_bias, (0, pad), "constant", 0)
    output = flow._C.fused_multi_head_attention_inference_v2(
        query=query,
        key=key,
        value=value,
        query_head_size=query_head_size,
        attn_mask_type=attn_mask_type,
        attn_bias=attn_bias,
        causal_diagonal_offset=causal_diagonal_offset,
        query_layout=query_layout,
        key_layout=key_layout,
        value_layout=value_layout,
        output_layout=output_layout,
        query_seq_start=query_seq_start,
        key_seq_start=key_seq_start,
        key_seq_len=key_seq_len.to(flow.int32).to("cuda") if use_kv_seq_len else None,
        query_max_seq_len=query_max_seq_len,
        key_max_seq_len=key_max_seq_len,
    )
    if output_layout == "BM(HK)" or output_layout == "(BM)(HK)":
        return output
    elif output_layout == "MB(HK)":
        return output.transpose(0, 1)
    else:
        raise NotImplementedError


def _test_fused_attention_concat_past_key_value(
    test_case,
    dtype,
    b,
    past_m,
    m,
    h,
    k,
    past_key_layout,
    past_value_layout,
    key_layout,
    value_layout,
):
    if past_m > 0:
        past_key = flow.randn((b, past_m, h, k), device="cuda", dtype=flow.float,).to(
            dtype
        )
        past_value = flow.randn((b, past_m, h, k), device="cuda", dtype=flow.float,).to(
            dtype
        )
    else:
        past_key = None
        past_value = None
    key = flow.randn((b, m, h, k), device="cuda", dtype=flow.float,).to(dtype)
    value = flow.randn((b, m, h, k), device="cuda", dtype=flow.float,).to(dtype)

    (
        fused_concated_key,
        fused_concated_value,
    ) = flow._C.fused_attention_concat_past_key_value(
        past_key=_to_layout([past_key, past_key, past_value], past_key_layout, 1),
        past_key_layout=past_key_layout,
        past_value=_to_layout([past_key, past_key, past_value], past_value_layout, 2),
        past_value_layout=past_value_layout,
        key=_to_layout([key, key, value], key_layout, 1),
        key_layout=key_layout,
        value=_to_layout([key, key, value], value_layout, 2),
        value_layout=value_layout,
        key_head_size=k,
    )
    if past_m > 0:
        concated_key = flow.cat([past_key, key], dim=1)
        concated_value = flow.cat([past_value, value], dim=1)
    else:
        concated_key = key
        concated_value = value
    ref_concated_key = _to_layout(
        [concated_key, concated_key, concated_value], past_key_layout, 1
    )
    ref_concated_value = _to_layout(
        [concated_key, concated_key, concated_value], past_value_layout, 2
    )
    test_case.assertTrue(
        np.array_equal(fused_concated_key.numpy(), ref_concated_key.numpy())
    )
    test_case.assertTrue(
        np.array_equal(fused_concated_value.numpy(), ref_concated_value.numpy())
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
    attn_mask_type="none",
    causal_diagonal_offset=0,
    query_layout="BM(HK)",
    key_layout="BM(HK)",
    value_layout="BM(HK)",
    output_layout="BM(HK)",
):
    query = flow.randn(
        (batch_size, query_seq_len, num_heads, query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    key = flow.randn(
        (batch_size, kv_seq_len, num_heads, query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    value = flow.randn(
        (batch_size, kv_seq_len, num_heads, value_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)

    fused_out = _fused_mha(
        query,
        key,
        value,
        num_heads,
        attn_mask_type=attn_mask_type,
        causal_diagonal_offset=causal_diagonal_offset,
        query_layout=query_layout,
        key_layout=key_layout,
        value_layout=value_layout,
        output_layout=output_layout,
    ).numpy()
    ref_out = _ref(
        query,
        key,
        value,
        num_heads,
        attn_mask_type=attn_mask_type,
        causal_diagonal_offset=causal_diagonal_offset,
    ).numpy()

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
    attn_mask_type="none",
):

    query = flow.randn(
        (batch_size, query_seq_len, num_heads, query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    key = flow.randn(
        (batch_size, kv_seq_len, num_heads, query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    value = flow.randn(
        (batch_size, kv_seq_len, num_heads, value_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)

    attn_bias = flow.randn((kv_seq_len,), device="cuda", dtype=flow.float).to(dtype)
    ref_out = _ref(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    fused_out = _fused_mha(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (query_seq_len, kv_seq_len), device="cuda", dtype=flow.float
    ).to(dtype)
    ref_out = _ref(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    fused_out = _fused_mha(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (num_heads, query_seq_len, kv_seq_len), device="cuda", dtype=flow.float
    ).to(dtype)
    ref_out = _ref(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    fused_out = _fused_mha(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (batch_size, num_heads, query_seq_len, kv_seq_len),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    ref_out = _ref(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    fused_out = _fused_mha(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))

    attn_bias = flow.randn(
        (num_heads, 1, kv_seq_len), device="cuda", dtype=flow.float
    ).to(dtype)
    ref_out = _ref(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    fused_out = _fused_mha(
        query, key, value, num_heads, attn_bias=attn_bias, attn_mask_type=attn_mask_type
    ).numpy()
    test_case.assertTrue(np.allclose(ref_out, fused_out, atol=1e-2, rtol=1e-2))


def _test_fused_multi_head_attention_inference_variable_length(
    test_case,
    batch_size,
    num_heads,
    query_seq_len,
    kv_seq_len,
    query_head_size,
    value_head_size,
    dtype,
    query_layout,
    key_layout,
    value_layout,
    use_kv_seq_len,
    attn_mask_type="none",
    causal_diagonal_offset=0,
):
    query = flow.randn(
        (batch_size, query_seq_len, num_heads, query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    key = flow.randn(
        (batch_size, kv_seq_len, num_heads, query_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)
    value = flow.randn(
        (batch_size, kv_seq_len, num_heads, value_head_size),
        device="cuda",
        dtype=flow.float,
    ).to(dtype)

    query_seq_len_t = flow.randint(
        low=1,
        high=query.shape[1],
        size=(query.shape[0],),
        device="cuda",
        dtype=flow.int32,
    )
    key_seq_len_t = flow.randint(
        low=1, high=key.shape[1], size=(key.shape[0],), device="cuda", dtype=flow.int32
    )

    fused_out = _fused_mha(
        query,
        key,
        value,
        num_heads,
        attn_mask_type=attn_mask_type,
        causal_diagonal_offset=causal_diagonal_offset,
        query_layout=query_layout,
        key_layout=key_layout,
        value_layout=value_layout,
        output_layout="(BM)(HK)",
        query_seq_len=query_seq_len_t,
        key_seq_len=key_seq_len_t,
        use_kv_seq_len=use_kv_seq_len,
    )
    ref_out = _ref(
        query,
        key,
        value,
        num_heads,
        attn_mask_type=attn_mask_type,
        causal_diagonal_offset=causal_diagonal_offset,
        query_seq_len=query_seq_len_t,
        key_seq_len=key_seq_len_t,
    )
    ref_out = ref_out.view(batch_size, query_seq_len, num_heads, value_head_size)
    ref_out = _to_layout([ref_out], "(BM)HK", 0, seq_len=query_seq_len_t)
    ref_out = ref_out.view(ref_out.shape[0], -1)

    test_case.assertTrue(
        np.allclose(ref_out.numpy(), fused_out.numpy(), atol=1e-2, rtol=1e-2)
    )


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
        _test_fused_multi_head_attention_inference(
            test_case,
            1,
            8,
            4,
            8,
            16,
            16,
            flow.float,
            attn_mask_type="causal_from_top_left",
            causal_diagonal_offset=4,
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
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float16, "causal_from_top_left"
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 4096, 40, 40, flow.float, "causal_from_bottom_right"
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float16
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float16, "causal_from_top_left"
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 80, 4096, 40, 40, flow.float16, "causal_from_bottom_right"
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 80, 40, 40, flow.float, "causal_from_top_left"
        )
        _test_fused_multi_head_attention_inference_with_attn_bias(
            test_case, 2, 8, 4096, 77, 40, 40, flow.float, "causal_from_top_left"
        )

    def test_multi_head_attention_inference_with_layout(test_case):
        layouts = [
            "BM(HK)",
            "BMHK",
            "MBHK",
            "BHMK",
            "MB(HK)",
            "BM(H3K)",
            "BM(H2K)",
            "MB(H3K)",
            "MB(H2K)",
        ]
        for query_layout, key_layout, value_layout in itertools.product(
            layouts, layouts, layouts
        ):
            if query_layout == "BM(H2K)" or query_layout == "MB(H2K)":
                continue
            _test_fused_multi_head_attention_inference(
                test_case,
                2,
                8,
                256,
                256,
                160,
                160,
                flow.float16,
                query_layout=query_layout,
                key_layout=key_layout,
                value_layout=value_layout,
            )

    def test_multi_head_attention_inference_with_output_layout(test_case):
        layouts = [
            "BM(HK)",
            "MB(HK)",
        ]
        for output_layout in layouts:
            _test_fused_multi_head_attention_inference(
                test_case,
                2,
                8,
                256,
                256,
                160,
                160,
                flow.float16,
                output_layout=output_layout,
            )
            _test_fused_multi_head_attention_inference(
                test_case,
                1,
                8,
                256,
                256,
                160,
                160,
                flow.float16,
                output_layout=output_layout,
            )

    def test_multi_head_attention_inference_variable_length(test_case):
        # test_case,batch_size, num_heads,query_seq_len, kv_seq_len,query_head_size,value_head_size,dtype
        layouts = ["(BM)HK", "(BM)(HK)", "(BM)(H2K)", "(BM)(H3K)"]
        for (
            query_layout,
            key_layout,
            value_layout,
            use_kv_seq_len,
        ) in itertools.product(layouts, layouts, layouts, (False, True)):
            if query_layout == "(BM)(H2K)":
                continue
            _test_fused_multi_head_attention_inference_variable_length(
                test_case,
                2,
                8,
                16,
                16,
                40,
                40,
                flow.float16,
                query_layout=query_layout,
                key_layout=key_layout,
                value_layout=value_layout,
                use_kv_seq_len=use_kv_seq_len,
            )
            if (
                query_layout == "(BM)(H3K)"
                or key_layout == "(BM)(H3K)"
                or value_layout == "(BM)(H3K)"
            ):
                continue
            _test_fused_multi_head_attention_inference_variable_length(
                test_case,
                2,
                8,
                16,
                32,
                40,
                40,
                flow.float16,
                query_layout=query_layout,
                key_layout=key_layout,
                value_layout=value_layout,
                use_kv_seq_len=use_kv_seq_len,
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestFusedAttentionConcatPastKeyValue(flow.unittest.TestCase):
    def test_fused_attention_concat_past_key_value(test_case):
        kv_layouts = [
            "BM(HK)",
            "BMHK",
            "MBHK",
            "BHMK",
            "MB(HK)",
            "BM(H3K)",
            # "BM(H2K)",
            # "MB(H3K)",
            "MB(H2K)",
        ]

        past_layouts = [
            "BM(HK)",
            "BMHK",
            # "MBHK",
            # "BHMK",
            "MB(HK)",
        ]

        types = [flow.float16]
        for (
            past_key_layout,
            past_value_layout,
            key_layout,
            value_layout,
            dtype,
        ) in itertools.product(
            past_layouts, past_layouts, kv_layouts, kv_layouts, types
        ):
            _test_fused_attention_concat_past_key_value(
                test_case,
                dtype,
                1,
                127,
                1,
                40,
                128,
                past_key_layout=past_key_layout,
                past_value_layout=past_value_layout,
                key_layout=key_layout,
                value_layout=value_layout,
            )
        _test_fused_attention_concat_past_key_value(
            test_case,
            flow.float,
            1,
            0,
            1,
            40,
            128,
            past_key_layout="BMHK",
            past_value_layout="BMHK",
            key_layout="BMHK",
            value_layout="BMHK",
        )


if __name__ == "__main__":
    unittest.main()
