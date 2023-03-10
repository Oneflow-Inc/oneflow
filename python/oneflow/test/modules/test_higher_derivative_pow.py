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


def _check_equal(test_case, lhs, rhs, rtol=1e-3, atol=1e-3):
    is_equal = np.allclose(
        lhs.detach().cpu().numpy(),
        rhs.detach().cpu().numpy(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    test_case.assertTrue(is_equal)


def _test_pow_grad_grad_impl(test_case):
    y_shape = [random().to(int).value() for _ in range(random().to(int).value())]
    x_shape = y_shape[random(0, 5).to(int).value() :]
    if random_bool().value():
        x_shape, y_shape = y_shape, x_shape

    # The range limit should be removed after solving issue #9908
    x = random_tensor(len(x_shape), *x_shape, low=0, high=1)
    y = random_tensor(len(y_shape), *y_shape, low=0, high=1)

    z = torch.pow(x, y)
    _check_equal(test_case, z.pytorch, z.oneflow)

    init_grad_z = random_tensor(len(z.oneflow.shape), *z.oneflow.shape)
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape)
    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape)

    dx_and_dy = torch.autograd.grad(z, [x, y], init_grad_z, True, True)
    _check_equal(test_case, dx_and_dy.pytorch[0], dx_and_dy.oneflow[0])
    _check_equal(test_case, dx_and_dy.pytorch[1], dx_and_dy.oneflow[1])

    ddx_ddy_ddz = torch.autograd.grad(
        dx_and_dy, [x, y, init_grad_z], [init_grad_x, init_grad_y]
    )
    _check_equal(test_case, ddx_ddy_ddz.pytorch[0], ddx_ddy_ddz.oneflow[0])
    _check_equal(test_case, ddx_ddy_ddz.pytorch[1], ddx_ddy_ddz.oneflow[1])
    _check_equal(test_case, ddx_ddy_ddz.pytorch[2], ddx_ddy_ddz.oneflow[2])


@flow.unittest.skip_unless_1n1d()
class TestPowHigherDerivative(flow.unittest.TestCase):
    def test_pow_grad_grad(test_case):
        for i in range(10):
            _test_pow_grad_grad_impl(test_case)


if __name__ == "__main__":
    unittest.main()
