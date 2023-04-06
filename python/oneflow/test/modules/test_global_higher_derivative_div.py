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


def _test_global_div_grad_grad_impl(test_case, placement):
    x_shape = [8, 8, 8, 8]
    y_shape = [8, 8]
    if random_bool().value():
        x_shape, y_shape = y_shape, x_shape
    x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    y = random_tensor(len(y_shape), *y_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    z = torch.div(x, y)
    init_grad_z = random_tensor(len(z.oneflow.shape), *z.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_x = random_tensor(len(x.oneflow.shape), *x.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )
    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=2)
    )

    dx_and_dy = torch.autograd.grad(z, [x, y], init_grad_z, True, True)
    test_case.assertTrue(
        np.allclose(
            dx_and_dy.pytorch[0].detach().cpu().numpy(),
            dx_and_dy.oneflow[0].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            dx_and_dy.pytorch[1].detach().cpu().numpy(),
            dx_and_dy.oneflow[1].detach().numpy(),
            rtol=1e-3,
            atol=1e-4,
        )
    )

    ddx_and_ddy_and_ddz = torch.autograd.grad(
        dx_and_dy, [x, y, init_grad_z], [init_grad_x, init_grad_y], True, True
    )
    test_case.assertTrue(
        np.allclose(
            ddx_and_ddy_and_ddz.pytorch[0].detach().cpu().numpy(),
            ddx_and_ddy_and_ddz.oneflow[0].detach().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
    )
    test_case.assertTrue(
        np.allclose(
            ddx_and_ddy_and_ddz.pytorch[1].detach().cpu().numpy(),
            ddx_and_ddy_and_ddz.oneflow[1].detach().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
    )
    test_case.assertTrue(
        np.allclose(
            ddx_and_ddy_and_ddz.pytorch[2].detach().cpu().numpy(),
            ddx_and_ddy_and_ddz.oneflow[2].detach().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
    )


class TestGlobalDivHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_global_div_grad_grad(test_case):
        for placement in all_placement():
            for i in range(1):
                _test_global_div_grad_grad_impl(test_case, placement)


if __name__ == "__main__":
    unittest.main()
