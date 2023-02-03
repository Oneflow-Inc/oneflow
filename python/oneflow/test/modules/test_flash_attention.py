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
from typing import List

import oneflow as flow
from oneflow.nn.functional import flash_attention


def gen_random_data(batch_size, num_heads, seqlen_q, seqlen_kv, head_dim, dtype):
    q = (
        flow.randn(batch_size, num_heads, seqlen_q, head_dim, requires_grad=True)
        .cuda()
        .to(dtype)
    )
    k = (
        flow.randn(batch_size, num_heads, seqlen_kv, head_dim, requires_grad=True)
        .cuda()
        .to(dtype)
    )
    v = (
        flow.randn(batch_size, num_heads, seqlen_kv, head_dim, requires_grad=True)
        .cuda()
        .to(dtype)
    )
    mask = flow.randn(batch_size, 1, 1, seqlen_kv).cuda().to(dtype)
    bias = (
        flow.randn(1, num_heads, seqlen_q, seqlen_kv, requires_grad=True)
        .cuda()
        .to(dtype)
    )
    return q, k, v, mask, bias


def _test_flash_attention(
    test_case,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_kv,
    head_dim,
    dtype,
    add_mask=True,
    add_bias=True,
    causal=False,
    dropout_p=0,
):
    q, k, v, mask, bias = gen_random_data(
        batch_size, num_heads, seqlen_q, seqlen_kv, head_dim, dtype
    )
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    bias.retain_grad()
    mask_01 = mask > 0.25  # bool, 0 or 1
    mask = ((1 - mask_01) * -10000.0).to(dtype)  # float, 0. or -inf
    out1 = flash_attention(q, k, v, mask=mask, bias=bias, unpad_kv=False)
    # skip test unpad_kv == True or unpad_kv == "auto"
    out2 = flash_attention(q, k, v, mask=mask, bias=bias, unpad_kv=False)
    out3 = flash_attention(q, k, v, mask=mask, bias=bias, unpad_kv=False)
    ref_out = flow.matmul(
        flow.softmax(
            flow.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5) + mask + bias,
            dim=-1,
        ),
        v,
    )

    atol, rtol = 0.009, 0.001
    if dtype == flow.bfloat16:
        atol, rtol = 0.09, 0.05
    test_case.assertTrue(
        np.allclose(out1.numpy(), ref_out.numpy(), atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(out2.numpy(), ref_out.numpy(), atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(out3.numpy(), ref_out.numpy(), atol=atol, rtol=rtol)
    )

    if dtype == flow.bfloat16: return
    # bwd
    ref_out.sum().backward()
    grad_q_ref, grad_k_ref, grad_v_ref, grad_bias_ref = (
        q.grad.numpy(),
        k.grad.numpy(),
        v.grad.numpy(),
        bias.grad.numpy(),
    )
    # out1 bwd
    q.grad, k.grad, v.grad, bias.grad = None, None, None, None
    out1.sum().backward()
    grad_q, grad_k, grad_v, grad_bias = (
        q.grad.numpy(),
        k.grad.numpy(),
        v.grad.numpy(),
        bias.grad.numpy(),
    )

    print(np.abs(grad_q-grad_q_ref).max())
    print(np.abs(grad_k-grad_k_ref).max())
    print(np.abs(grad_v-grad_v_ref).max())
    print(np.abs(grad_bias-grad_bias_ref).max())
    test_case.assertTrue(
        np.allclose(grad_q, grad_q_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_k, grad_k_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_v, grad_v_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_bias, grad_bias_ref, atol=atol, rtol=rtol)
    )
    del out1, grad_q, grad_k, grad_v, grad_bias

    # out2 bwd
    q.grad, k.grad, v.grad, bias.grad = None, None, None, None
    out2.sum().backward()
    grad_q, grad_k, grad_v, grad_bias = (
        q.grad.numpy(),
        k.grad.numpy(),
        v.grad.numpy(),
        bias.grad.numpy(),
    )

    test_case.assertTrue(
        np.allclose(grad_q, grad_q_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_k, grad_k_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_v, grad_v_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_bias, grad_bias_ref, atol=atol, rtol=rtol)
    )
    del out2, grad_q, grad_k, grad_v, grad_bias

    # out3 bwd
    q.grad, k.grad, v.grad, bias.grad = None, None, None, None
    out3.sum().backward()
    grad_q, grad_k, grad_v, grad_bias = (
        q.grad.numpy(),
        k.grad.numpy(),
        v.grad.numpy(),
        bias.grad.numpy(),
    )

    test_case.assertTrue(
        np.allclose(grad_q, grad_q_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_k, grad_k_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_v, grad_v_ref, atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(grad_bias, grad_bias_ref, atol=atol, rtol=rtol)
    )
    del out3, grad_q, grad_k, grad_v, grad_bias


@flow.unittest.skip_unless_1n1d()
class TestFlashAttention(flow.unittest.TestCase):
    def test_flash_attention(test_case):
        # batch_size, num_heads, seqlen_q, seqlen_kv, head_dim, dtype

        _test_flash_attention(test_case, 64, 4, 20, 23, 32, flow.float16)
        _test_flash_attention(test_case, 32, 8, 32, 32, 16, flow.float16)
        _test_flash_attention(test_case, 16, 8, 512, 512, 32, flow.float16)
        _test_flash_attention(test_case, 8, 8, 2022, 2023, 32, flow.float16)

        _test_flash_attention(test_case, 64, 4, 20, 23, 80, flow.bfloat16)
        _test_flash_attention(test_case, 32, 8, 32, 32, 16, flow.bfloat16)
        _test_flash_attention(test_case, 16, 8, 512, 512, 32, flow.bfloat16)
        _test_flash_attention(test_case, 8, 8, 2022, 2023, 32, flow.bfloat16)


if __name__ == "__main__":
    unittest.main()
