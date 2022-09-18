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


def _test_avg_pool_grad_grad_impl(test_case, placement, ndim):
    x_shape = [8, 8] + [5] * ndim

    m = eval(f"torch.nn.AvgPool{ndim}d")(kernel_size=random(2, 5).to(int))

    x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    y = m(x)
    _check_equal(test_case, y.pytorch, y.oneflow, "y")

    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, "dx")

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x, True, True)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]
    _check_equal(test_case, ddx.pytorch, ddx.oneflow, "ddx")
    _check_equal(test_case, ddy.pytorch, ddy.oneflow, "ddy")


def _test_max_pool_grad_grad_impl(test_case, placement, ndim):
    x_shape = [8, 8] + [5] * ndim

    m = eval(f"torch.nn.MaxPool{ndim}d")(kernel_size=random(2, 5).to(int))

    x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )

    y = m(x)
    _check_equal(test_case, y.pytorch, y.oneflow, "y")

    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, "dx")

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x, True, True)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]
    _check_equal(test_case, ddx.pytorch, ddx.oneflow, "ddx")
    _check_equal(test_case, ddy.pytorch, ddy.oneflow, "ddy")


def _test_adaptive_pool_grad_grad_impl(test_case, placement, ndim, mode):
    x_shape = [8, 8] + [5] * ndim

    m = eval(f"torch.nn.Adaptive{mode.title()}Pool{ndim}d")(
        output_size=random(2, 5).to(int)
    )

    x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    y = m(x)
    _check_equal(test_case, y.pytorch, y.oneflow, "y")

    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _check_equal(test_case, dx.pytorch, dx.oneflow, "dx")

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x, True, True)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]

    _check_equal(test_case, ddx.pytorch, ddx.oneflow, "ddx")
    _check_equal(test_case, ddy.pytorch, ddy.oneflow, "ddy")


class TestGlobalPoolHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_max_pool_1d_grad_grad(test_case):
        for placement in all_placement():
            _test_max_pool_grad_grad_impl(test_case, placement, 1)

    @globaltest
    def test_max_pool_2d_grad_grad(test_case):
        for placement in all_placement():
            _test_max_pool_grad_grad_impl(test_case, placement, 2)

    @globaltest
    def test_max_pool_3d_grad_grad(test_case):
        for placement in all_placement():
            _test_max_pool_grad_grad_impl(test_case, placement, 3)

    @globaltest
    def test_avg_pool_1d_grad_grad(test_case):
        for placement in all_placement():
            _test_avg_pool_grad_grad_impl(test_case, placement, ndim=1)

    @globaltest
    def test_avg_pool_2d_grad_grad(test_case):
        for placement in all_placement():
            _test_avg_pool_grad_grad_impl(test_case, placement, ndim=2)

    @globaltest
    def test_avg_pool_3d_grad_grad(test_case):
        for placement in all_placement():
            _test_avg_pool_grad_grad_impl(test_case, placement, ndim=3)

    @globaltest
    def test_adaptive_avg_pool_1d_grad_grad(test_case):
        for placement in all_placement():
            _test_adaptive_pool_grad_grad_impl(test_case, placement, ndim=1, mode="avg")

    @globaltest
    def test_adaptive_avg_pool_2d_grad_grad(test_case):
        for placement in all_placement():
            _test_adaptive_pool_grad_grad_impl(test_case, placement, ndim=2, mode="avg")

    @globaltest
    def test_adaptive_avg_pool_3d_grad_grad(test_case):
        for placement in all_placement():
            _test_adaptive_pool_grad_grad_impl(test_case, placement, ndim=3, mode="avg")


if __name__ == "__main__":
    unittest.main()
