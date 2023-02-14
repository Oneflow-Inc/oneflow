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
import os
from einops import repeat

import oneflow as flow
from oneflow.nn.functional import flash_attention


def gen_bert_mask(k):
    # 0001111111110000
    b, s = k.shape[-4], k.shape[-2]
    se = flow.randint(1, s - 1, (b, 2)).cuda()
    start = se.min(dim=-1) - 1
    end = se.max(dim=-1) + 1
    padding_mask = repeat(flow.arange(s).cuda(), "s -> b s", b=b)
    padding_mask = (padding_mask >= start) + (padding_mask < end)
    return padding_mask


def gen_random_data(
    batch_size, num_heads, seqlen_q, seqlen_kv, head_dim, dtype, padding="random"
):
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
    if padding == "bert":
        padding = gen_bert_mask(k)
    else:
        mask = flow.randn(batch_size, 1, 1, seqlen_kv).cuda().to(dtype) > -0.1
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
    padding="random",
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
    out1 = flash_attention(q, k, v, mask=mask, bias=bias, unpad_kv=False)
    if dtype == flow.bfloat16:
        out2 = out1  # dim_gather do not support bf16
    else:
        out2 = flash_attention(q, k, v, mask=mask, bias=bias, unpad_kv=True)
    out3 = flash_attention(q, k, v, mask=mask, bias=bias, unpad_kv="auto")

    mask_fp = ((1 - mask) * -10000.0).to(dtype)  # float, 0. or -inf
    ref_out = flow.matmul(
        flow.softmax(
            flow.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5) + mask_fp + bias,
            dim=-1,
        ),
        v,
    )

    atol, rtol = 0.009, 0.005
    if dtype == flow.bfloat16:
        atol, rtol = 0.05, 0.01
    test_case.assertTrue(
        np.allclose(out1.numpy(), ref_out.numpy(), atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(out2.numpy(), ref_out.numpy(), atol=atol, rtol=rtol)
    )
    test_case.assertTrue(
        np.allclose(out3.numpy(), ref_out.numpy(), atol=atol, rtol=rtol)
    )

    # sm75 && sm86 do not have enough shared memory when head_dim>64
    dprop = flow.cuda.get_device_properties()
    is_sm86 = dprop.major == 8 and dprop.minor == 6
    is_sm75 = dprop.major == 7 and dprop.minor == 5
    if dtype == flow.bfloat16 or ((is_sm86 or is_sm75) and head_dim > 64):
        return
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

    test_case.assertTrue(np.allclose(grad_q, grad_q_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_k, grad_k_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_v, grad_v_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_bias, grad_bias_ref, atol=atol, rtol=rtol))
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

    test_case.assertTrue(np.allclose(grad_q, grad_q_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_k, grad_k_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_v, grad_v_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_bias, grad_bias_ref, atol=atol, rtol=rtol))
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

    test_case.assertTrue(np.allclose(grad_q, grad_q_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_k, grad_k_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_v, grad_v_ref, atol=atol, rtol=rtol))
    test_case.assertTrue(np.allclose(grad_bias, grad_bias_ref, atol=atol, rtol=rtol))
    del out3, grad_q, grad_k, grad_v, grad_bias


@unittest.skipIf(True, "skip test")
@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFlashAttention(flow.unittest.TestCase):
    def test_flash_attention(test_case):
        # batch_size, num_heads, seqlen_q, seqlen_kv, head_dim, dtype
        arg_dict = OrderedDict()
        arg_dict["test_function"] = [_test_flash_attention]
        arg_dict["batch_size"] = [8, 128, 512]
        arg_dict["num_heads"] = [4, 8]
        arg_dict["seqlen_q"] = [32, 256, 1024, 2500]
        arg_dict["seqlen_kv"] = [128, 333, 1501, 3072]
        arg_dict["head_dim"] = [40, 64, 128]
        arg_dict["dtype"] = [flow.float16, flow.bfloat16]
        arg_dict["padding"] = ["random", "bert"]
        for arg in GenArgList(arg_dict):
            print(arg[1:])
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
