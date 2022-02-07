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
import os

import numpy as np
from test_util import GenArgList

import oneflow as flow


def _test_fused_matmul_bias_add_relu(test_case, transpose_a, transpose_b, dtype):
    m = np.random.randint(low=1, high=128)
    k = np.random.randint(low=1, high=128)
    n = np.random.randint(low=1, high=128)
    alpha = np.random.random()

    a = np.random.randn(m, k)
    if transpose_a:
        a = np.transpose(a, (1, 0))

    b = np.random.randn(k, n)
    if transpose_b:
        b = np.transpose(b, (1, 0))

    bias = np.random.randn(n)

    fused_a_tensor = flow.tensor(a, dtype=dtype, device="cuda")
    fused_b_tensor = flow.tensor(b, dtype=dtype, device="cuda")
    fused_bias_tensor = flow.tensor(bias, dtype=dtype, device="cuda")
    fused_a_tensor.requires_grad = True
    fused_b_tensor.requires_grad = True
    fused_bias_tensor.requires_grad = True

    fused_out = flow._C.fused_matmul_bias_add_relu(
        fused_a_tensor,
        fused_b_tensor,
        fused_bias_tensor,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        alpha=alpha,
    )

    origin_a_tensor = flow.tensor(a, dtype=dtype, device="cuda")
    origin_b_tensor = flow.tensor(b, dtype=dtype, device="cuda")
    origin_bias_tensor = flow.tensor(bias, dtype=dtype, device="cuda")

    origin_a_tensor.requires_grad = True
    origin_b_tensor.requires_grad = True
    origin_bias_tensor.requires_grad = True

    origin_out = flow._C.relu(
        flow._C.bias_add(
            flow._C.matmul(
                origin_a_tensor,
                origin_b_tensor,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
            ),
            origin_bias_tensor,
            axis=1,
        )
    )  # TODO: currently only support 2d fused matmul.

    total_out = fused_out.sum() + origin_out.sum()
    total_out.backward()

    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(
            fused_a_tensor.grad.numpy(),
            origin_a_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_b_tensor.grad.numpy(),
            origin_b_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedMatmulBiasAddRelu(flow.unittest.TestCase):
    def test_fused_matmul_op(test_case):
        args_dict = OrderedDict()
        args_dict["test_fun"] = [_test_fused_matmul_bias_add_relu]
        args_dict["transpose_a"] = [False, True]
        args_dict["transpose_b"] = [False, True]
        args_dict["dtype"] = [flow.float32, flow.float64]

        for arg in GenArgList(args_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
