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
import os

import oneflow as flow


def _scaled_dot_product_attention(
    query, key, value,
):
    # input dims will equal 3 or 4.
    if key.ndim == 4:
        key = key.permute(0, 1, 3, 2)
    elif key.ndim == 3:
        key = key.permute(0, 2, 1)
    scores = flow.matmul(query, key) / math.sqrt(query.shape[-1])
    attn = flow.softmax(scores, dim=-1)
    out = flow.matmul(attn, value)
    return out


def _test_scaled_dot_product_attention(
    test_case, batch_size, num_head_pair, seq_len_pair, head_size, dtype,
):
    num_heads = num_head_pair[0]
    num_heads_k = num_head_pair[1]
    seq_len_q = seq_len_pair[0]
    seq_len_kv = seq_len_pair[1]
    query_raw = np.random.uniform(
        low=-1, high=1, size=(batch_size, num_heads, seq_len_q, head_size)
    )
    key_raw = np.random.uniform(
        low=-1, high=1, size=(batch_size, num_heads_k, seq_len_kv, head_size)
    )
    value_raw = np.random.uniform(
        low=-1, high=1, size=(batch_size, num_heads_k, seq_len_kv, head_size)
    )
    query_fused = flow.tensor(query_raw, dtype=dtype, device="cuda", requires_grad=True)
    query_ref = flow.tensor(query_raw, dtype=dtype, device="cuda", requires_grad=True)
    key_fused = flow.tensor(key_raw, dtype=dtype, device="cuda", requires_grad=True)
    key_ref = flow.tensor(key_raw, dtype=dtype, device="cuda", requires_grad=True)
    value_fused = flow.tensor(value_raw, dtype=dtype, device="cuda", requires_grad=True)
    value_ref = flow.tensor(value_raw, dtype=dtype, device="cuda", requires_grad=True)

    fused_out = flow._C.scaled_dot_product_attention(
        query=query_fused, key=key_fused, value=value_fused,
    )
    if num_heads == num_heads_k:
        ref_out = _scaled_dot_product_attention(query_ref, key_ref, value_ref,)
    else:  # For GQA
        ref_out = flow.empty(query_fused.shape, device="cuda", dtype=dtype)
        stride = num_heads / num_heads_k
        for i in range(0, num_heads):
            j = int(i / stride)
            ref_out[:, i, :, :] = _scaled_dot_product_attention(
                query_ref[:, i, :, :], key_ref[:, j, :, :], value_ref[:, j, :, :]
            )

    total_out = ref_out.sum() + fused_out.sum()
    total_out.backward()
    if dtype == flow.float16:
        error_tol = 1.0
    elif dtype == flow.bfloat16:
        error_tol = 1.0
    else:
        error_tol = 1e-3

    test_case.assertTrue(
        np.allclose(ref_out.numpy(), fused_out.numpy(), atol=error_tol, rtol=error_tol)
    )
    test_case.assertTrue(
        np.allclose(
            query_fused.grad.numpy(),
            query_ref.grad.numpy(),
            atol=error_tol,
            rtol=error_tol,
        )
    )
    test_case.assertTrue(
        np.allclose(
            key_fused.grad.numpy(), key_ref.grad.numpy(), atol=error_tol, rtol=error_tol
        )
    )
    test_case.assertTrue(
        np.allclose(
            value_fused.grad.numpy(),
            value_ref.grad.numpy(),
            atol=error_tol,
            rtol=error_tol,
        )
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestScaledDotProductAttention(flow.unittest.TestCase):
    def test_scaled_dot_product_attention(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_scaled_dot_product_attention]
        args_dict["batchsize"] = [1, 2, 4]
        args_dict["num_head_pair"] = [[16, 16], [16, 8]]
        args_dict["seqlen_pair"] = [[4096, 4096], [4096, 77], [1024, 1024], [1024, 77]]
        args_dict["head_size"] = [40, 80, 160, 41]
        args_dict["dtype"] = [flow.float16]

        if flow._oneflow_internal.flags.with_cuda():
            if flow._oneflow_internal.flags.cuda_version() >= 11070:
                if flow.cuda.get_device_capability()[0] >= 8:
                    for arg in GenArgList(args_dict):
                        arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
