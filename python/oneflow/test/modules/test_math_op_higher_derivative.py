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
    x = random_tensor(ndim=2).requires_grad_(True)
    y = eval(f"x.{op_name}().sum()")
    x_grad = torch.autograd.grad(y, x, create_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            x_grad.pytorch.detach().cpu().numpy(), x_grad.oneflow.detach().numpy()
        )
    )
    x_grad_grad = torch.autograd.grad(x_grad, x, torch.ones_like(x), create_graph=True)[
        0
    ]
    test_case.assertTrue(
        np.allclose(
            x_grad_grad.pytorch.detach().cpu().numpy(),
            x_grad_grad.oneflow.detach().numpy(),
        )
    )

def _test_math_binary_op_grad_grad_impl(test_case, op_name):
    dim1 = random(2, 6)
    x = random_tensor(ndim=2, dim1=dim1).requires_grad_(True)
    y = random_tensor(ndim=2, dim1=dim1).requires_grad_(True)
    z = eval(f"x.{op_name}(y).sum()")
    x_grad = torch.autograd.grad(z, x, create_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            x_grad.pytorch.detach().cpu().numpy(), x_grad.oneflow.detach().numpy()
        )
    )

    y_grad = torch.autograd.grad(z, y, create_graph=True)[0]
    test_case.assertTrue(
        np.allclose(
            y_grad.pytorch.detach().cpu().numpy(), y_grad.oneflow.detach().numpy()
        )
    )

    x_grad_x_grad = torch.autograd.grad(y_grad, x, torch.ones_like(x), create_graph=True)[0]
    
    test_case.assertTrue(
        np.allclose(
            x_grad_x_grad.pytorch.detach().cpu().numpy(),
            x_grad_x_grad.oneflow.detach().numpy(),
        )
    )



class TestMathOpHigherDerivative(flow.unittest.TestCase):
    def test_sin_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "sin")

    def test_cos_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "cos")

    def test_exp_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "exp")

    def test_log_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "log")
    
    def test_pow_grad_grad(test_case):
        _test_math_binary_op_grad_grad_impl(test_case, "pow")

    def test_tanh_grad_grad(test_case):
        _test_math_op_grad_grad_impl(test_case, "tanh")


if __name__ == "__main__":
    unittest.main()
