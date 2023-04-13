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


def _assert_true(test_case, value1, value2):
    test_case.assertTrue(
        np.allclose(
            value1.detach().cpu().numpy(),
            value2.detach().cpu().numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
    )


def _test_softmax_grad_grad_impl(test_case, op_name):
    ndim = random(low=2).to(int).value()
    data = random_tensor(ndim=ndim)

    for dim in range(ndim):
        x = data.detach().clone().requires_grad_()
        m = eval(f"torch.nn.{op_name}")(dim)
        y = m(x)
        _assert_true(test_case, y.pytorch, y.oneflow)

        x_shape = x.oneflow.shape
        init_grad_x = random_tensor(len(x_shape), *x_shape)
        init_grad_y = random_tensor(len(x_shape), *x_shape)

        dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
        _assert_true(test_case, dx.pytorch, dx.oneflow)

        ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x)
        ddx, ddy = ddx_ddy[0], ddx_ddy[1]
        _assert_true(test_case, ddx.pytorch, ddx.oneflow)
        _assert_true(test_case, ddy.pytorch, ddy.oneflow)


@flow.unittest.skip_unless_1n1d()
class TestSoftmaxHigherDerivative(flow.unittest.TestCase):
    def test_softmax_grad_grad(test_case):
        _test_softmax_grad_grad_impl(test_case, op_name="Softmax")

    def test_logsoftmax_grad_grad(test_case):
        _test_softmax_grad_grad_impl(test_case, op_name="LogSoftmax")


if __name__ == "__main__":
    unittest.main()
