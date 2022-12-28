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


def _check_equal(test_case, lhs, rhs, rtol=1e-4, atol=1e-4, name=""):
    is_equal = np.allclose(
        lhs.detach().cpu().numpy(),
        rhs.detach().cpu().numpy(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    test_case.assertTrue(is_equal, f"{name} is not equal")


def _test_scalar_pow_grad_grad_impl(test_case, reverse=False):
    x_shape = [random().to(int).value() for _ in range(random().to(int).value())]
    y = random().to(float if random_bool().value() else int).value()

    x = random_tensor(len(x_shape), *x_shape)
    z = torch.pow(x, y) if not reverse else torch.pow(y, x)

    init_grad_z = random_tensor(len(z.oneflow.shape), *z.oneflow.shape)
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape)

    dx = torch.autograd.grad(z, x, init_grad_z, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, name="dx")

    ddx_and_ddz = torch.autograd.grad(dx, [x, init_grad_z], init_grad_x, True, True)
    _check_equal(test_case, ddx_and_ddz.pytorch[0], ddx_and_ddz.oneflow[0], name="ddx")
    _check_equal(test_case, ddx_and_ddz.pytorch[1], ddx_and_ddz.oneflow[1], name="ddz")


class TestScalarPowHigherDerivative(flow.unittest.TestCase):
    def test_scalar_pow_grad_grad(test_case):
        for i in range(10):
            _test_scalar_pow_grad_grad_impl(test_case)

    def test_scalar_reverse_pow_grad_grad(test_case):
        for i in range(10):
            _test_scalar_pow_grad_grad_impl(test_case, reverse=True)


if __name__ == "__main__":
    unittest.main()
