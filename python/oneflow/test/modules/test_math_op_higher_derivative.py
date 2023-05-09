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

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def _test_math_op_grad_grad_impl(test_case, op_name):
    x = random_tensor(ndim=2, low=-2, high=2).requires_grad_(True)
    y = eval(f"torch.{op_name}")(x)
    np_arr = np.random.rand(*x.oneflow.shape)
    init_grad = torch.tensor(np_arr).requires_grad_()

    x_grad = torch.autograd.grad(y, x, init_grad, retain_graph=True, create_graph=True)[
        0
    ]
    test_case.assertTrue(
        np.allclose(
            x_grad.pytorch.detach().cpu().numpy(),
            x_grad.oneflow.detach().numpy(),
            atol=1e-4,
            rtol=1e-4,
            equal_nan=True,
        )
    )

    x_grad_grad = torch.autograd.grad(x_grad, x, init_grad, retain_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            x_grad_grad.pytorch.detach().cpu().numpy(),
            x_grad_grad.oneflow.detach().numpy(),
            atol=1e-4,
            rtol=1e-4,
            equal_nan=True,
        )
    )

    init_grad_grad = torch.tensor(np_arr).requires_grad_()
    dgrad = torch.autograd.grad(x_grad, init_grad, init_grad_grad, retain_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            dgrad.pytorch.detach().cpu().numpy(),
            dgrad.oneflow.detach().numpy(),
            atol=1e-4,
            rtol=1e-4,
            equal_nan=True,
        )
    )


class TestMathOpHigherDerivative(flow.unittest.TestCase):
    def test_sin_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "sin")

    def test_cos_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "cos")

    def test_tan_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "tan")

    def test_sinh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "sinh")

    def test_cosh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "cosh")

    def test_tanh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "tanh")

    def test_asin_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "asin")

    def test_acos_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "acos")

    def test_atan_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "atan")

    def test_asinh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "asinh")

    def test_acosh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "acosh")

    def test_atanh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "atanh")

    def test_erf_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "erf")

    def test_erfc_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "erfc")

    def test_exp_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "exp")

    def test_exp2_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "exp2")

    def test_expm1_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "expm1")

    def test_log_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "log")

    def test_logsigmoid_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "nn.functional.logsigmoid")

    def test_log2_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "log2")

    def test_log1p_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "log1p")

    def test_reciprocal_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "reciprocal")

    def test_rsqrt_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "rsqrt")

    def test_sqrt_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "sqrt")

    def test_square_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "square")

    def test_sigmoid_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "sigmoid")

    def test_abs_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "abs")


if __name__ == "__main__":
    unittest.main()
