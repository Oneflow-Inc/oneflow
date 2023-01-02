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


def _check_equal(test_case, lhs, rhs, name="", rtol=1e-5, atol=1e-5):
    is_equal = np.allclose(
        lhs.detach().cpu().numpy(),
        rhs.detach().cpu().numpy(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    test_case.assertTrue(is_equal, f"{name} is not equal" if name else "")


def _test_avg_pool_grad_grad_impl(test_case, ndim):
    device = random_device()
    minibatch = random(1, 5).to(int).value()
    channels = random(1, 5).to(int).value()
    padding = random(0, 3).to(int).value()
    ceil_mode = random_bool().value()
    count_include_pad = random_bool().value()
    divisor_override = random().to(int).value()
    kernel_size = random(4, 6).to(int).value()
    stride = random(1, 3).to(int).value()
    x_shape = [minibatch, channels] + [
        random(8, 12).to(int).value() for i in range(ndim)
    ]

    kwargs = {
        "kernel_size": kernel_size,
        "stride": oneof(stride, nothing()),
        "padding": oneof(padding, nothing()),
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    if ndim != 1:
        kwargs["divisor_override"] = divisor_override

    m = eval(f"torch.nn.AvgPool{ndim}d")(**kwargs)
    m.to(device)

    x = random_tensor(len(x_shape), *x_shape).to(device)
    y = m(x)
    _check_equal(test_case, y.pytorch, y.oneflow, "y")

    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to(device)
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to(device)

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, "dx")

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x, True, True)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]
    _check_equal(test_case, ddx.pytorch, ddx.oneflow, "ddx")
    _check_equal(test_case, ddy.pytorch, ddy.oneflow, "ddy")


def _test_max_pool_grad_grad_impl(test_case, ndim):
    device = random_device()
    minibatch = random(1, 5).to(int).value()
    channels = random(1, 5).to(int).value()
    padding = random(0, 3).to(int).value()
    dilation = random(1, 3).to(int).value()
    ceil_mode = random_bool().value()
    return_indices = random_bool().value()
    kernel_size = random(4, 6).to(int).value()
    stride = random(1, 3).to(int).value()
    x_shape = [minibatch, channels] + [
        random(10, 12).to(int).value() for i in range(ndim)
    ]

    m = eval(f"torch.nn.MaxPool{ndim}d")(
        kernel_size=kernel_size,
        stride=oneof(stride, nothing()),
        padding=oneof(padding, nothing()),
        dilation=oneof(dilation, nothing()),
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )
    m.to(device)

    x = random_tensor(len(x_shape), *x_shape).to(device)
    if return_indices:
        y_and_indices = m(x)
        y, indices = y_and_indices[0], y_and_indices[1]
    else:
        y = m(x)
    _check_equal(test_case, y.pytorch, y.oneflow, "y")

    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to(device)
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to(device)

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, "dx")

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x, True, True)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]
    _check_equal(test_case, ddx.pytorch, ddx.oneflow, "ddx")
    _check_equal(test_case, ddy.pytorch, ddy.oneflow, "ddy")


def _test_adaptive_pool_grad_grad_impl(test_case, ndim, mode):
    device = random_device()
    x_shape = [random(5, 10).to(int).value() for i in range(2 + ndim)]
    output_size = [random(2, 1 + x_shape[2 + i]).to(int).value() for i in range(ndim)]

    m = eval(f"torch.nn.Adaptive{mode.title()}Pool{ndim}d")(output_size)
    m.to(device)

    x = random_tensor(len(x_shape), *x_shape).to(device)
    y = m(x)
    _check_equal(test_case, y.pytorch, y.oneflow, "y")

    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to(device)
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to(device)

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, "dx")

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x, True, True)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]

    _check_equal(test_case, ddx.pytorch, ddx.oneflow, "ddx")
    _check_equal(test_case, ddy.pytorch, ddy.oneflow, "ddy")


class TestPoolHigherDerivative(flow.unittest.TestCase):
    def test_max_pool_1d_grad_grad(test_case):
        _test_max_pool_grad_grad_impl(test_case, 1)

    def test_max_pool_2d_grad_grad(test_case):
        _test_max_pool_grad_grad_impl(test_case, 2)

    def test_max_pool_3d_grad_grad(test_case):
        _test_max_pool_grad_grad_impl(test_case, 3)

    def test_avg_pool_1d_grad_grad(test_case):
        _test_avg_pool_grad_grad_impl(test_case, ndim=1)

    def test_avg_pool_2d_grad_grad(test_case):
        _test_avg_pool_grad_grad_impl(test_case, ndim=2)

    def test_avg_pool_3d_grad_grad(test_case):
        _test_avg_pool_grad_grad_impl(test_case, ndim=3)

    def test_adaptive_avg_pool_1d_grad_grad(test_case):
        _test_adaptive_pool_grad_grad_impl(test_case, ndim=1, mode="avg")

    def test_adaptive_avg_pool_2d_grad_grad(test_case):
        _test_adaptive_pool_grad_grad_impl(test_case, ndim=2, mode="avg")

    def test_adaptive_avg_pool_3d_grad_grad(test_case):
        _test_adaptive_pool_grad_grad_impl(test_case, ndim=3, mode="avg")


if __name__ == "__main__":
    unittest.main()
