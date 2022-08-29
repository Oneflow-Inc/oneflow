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


def _global_math_op_grad_grad_impl(test_case, op_name, placement, sbp):
    x = (
        random_tensor(2, dim0=8, dim1=8, low=-2, high=2)
        .to_global(placement=placement, sbp=sbp)
        .requires_grad_(True)
    )
    y = eval(f"x.{op_name}()")
    init_grad = random_tensor(2, 8, 8).to_global(placement, sbp).requires_grad_()

    x_grad = torch.autograd.grad(y, x, init_grad, create_graph=True)[0]
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

    init_grad_grad = random_tensor(2, 8, 8).to_global(placement, sbp).requires_grad_()
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


class TestGlobalMathOpHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_global_sin_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "sin", placement, sbp)

    @globaltest
    def test_global_cos_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "cos", placement, sbp)

    @globaltest
    def test_global_tan_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "tan", placement, sbp)

    @globaltest
    def test_global_sinh_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "sinh", placement, sbp)

    @globaltest
    def test_global_cosh_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "cosh", placement, sbp)

    @globaltest
    def test_global_tanh_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "tanh", placement, sbp)

    @globaltest
    def test_global_asin_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "asin", placement, sbp)

    @globaltest
    def test_global_acos_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "acos", placement, sbp)

    @globaltest
    def test_global_atan_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "atan", placement, sbp)

    @globaltest
    def test_global_asinh_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "asinh", placement, sbp)

    @globaltest
    def test_global_acosh_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "acosh", placement, sbp)

    @globaltest
    def test_global_atanh_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _global_math_op_grad_grad_impl(test_case, "atanh", placement, sbp)


if __name__ == "__main__":
    unittest.main()
