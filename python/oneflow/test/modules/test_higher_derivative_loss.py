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
    # print('='*80)
    # print(value1, value2)
    test_case.assertTrue(
        np.allclose(
            value1.detach().cpu().numpy(),
            value2.detach().numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
    )


def generate_necessity_for_cross_entropy_or_nll_loss(dim: int):
    if dim > 5 or dim < 2:
        raise ValueError("dim should be less than 5 or greater than 1. ")
    device = random_device()
    num_classes = random(low=2).to(int)
    batch_size = random(low=10, high=100).to(int)
    ignore_index = (
        random(0, num_classes).to(int) | nothing()
        if num_classes.value() > 2
        else nothing()
    )
    extra_dim = [random().to(int) for _ in range(dim - 2)]
    return (
        random_tensor(dim, batch_size, num_classes, *extra_dim).to(device),
        random_tensor(
            dim - 1,
            batch_size,
            *extra_dim,
            low=0,
            high=num_classes,
            dtype=int,
            requires_grad=False,
        ).to(device),
        random_tensor(1, num_classes, low=0, high=3, requires_grad=False).to(device),
        ignore_index,
        device,
    )


def _test_smooth_l1_loss_grad_grad_impl(test_case):
    device = random_device()
    x_shape = random_tensor().oneflow.shape

    x = random_tensor(len(x_shape), *x_shape).to(device)
    y = random_tensor(len(x_shape), *x_shape, requires_grad=False).to(device)
    m = torch.nn.SmoothL1Loss(
        reduction=oneof("none", "sum", "mean", nothing()), beta=oneof(0.0, 0.5, 1)
    )
    m.to(device)

    z = m(x, y)
    z_shape = z.oneflow.shape
    _assert_true(test_case, z.pytorch, z.oneflow)

    x_shape = x.oneflow.shape
    init_grad_z = random_tensor(len(z_shape), *z_shape).to(device)
    init_grad_x = random_tensor(len(x_shape), *x_shape).to(device)

    dx = torch.autograd.grad(z, x, init_grad_z, True, True)[0]
    _assert_true(test_case, dx.pytorch, dx.oneflow)

    ddx_ddz = torch.autograd.grad(dx, [x, init_grad_z], init_grad_x)
    ddx, ddz = ddx_ddz[0], ddx_ddz[1]
    _assert_true(test_case, ddx.pytorch, ddx.oneflow)
    _assert_true(test_case, ddz.pytorch, ddz.oneflow)


def _test_nll_loss_grad_grad_impl(test_case):
    (
        x,
        target,
        weight,
        ignore_index,
        device,
    ) = generate_necessity_for_cross_entropy_or_nll_loss(3)
    m = torch.nn.NLLLoss(
        weight=oneof(weight, nothing()),
        reduction=oneof("none", "sum", "mean", nothing()),
        ignore_index=ignore_index,
    )
    m.to(device)

    z = m(x, target)
    z_shape = z.oneflow.shape
    _assert_true(test_case, z.pytorch, z.oneflow)

    x_shape = x.oneflow.shape
    init_grad_z = random_tensor(len(z_shape), *z_shape).to(device)
    init_grad_x = random_tensor(len(x_shape), *x_shape).to(device)

    dx = torch.autograd.grad(z, x, init_grad_z, True, True)[0]
    _assert_true(test_case, dx.pytorch, dx.oneflow)

    ddx_ddz = torch.autograd.grad(dx, [x, init_grad_z], init_grad_x)
    ddx, ddz = ddx_ddz[0], ddx_ddz[1]
    _assert_true(test_case, ddx.pytorch, ddx.oneflow)
    _assert_true(test_case, ddz.pytorch, ddz.oneflow)


class TestLossHigherDerivative(flow.unittest.TestCase):
    def test_smooth_l1_loss_grad_grad(test_case):
        for i in range(5):
            print(".", end="", flush=True)
            _test_smooth_l1_loss_grad_grad_impl(test_case)

    def test_nll_loss_grad_grad(test_case):
        for i in range(5):
            print(".", end="", flush=True)
            _test_nll_loss_grad_grad_impl(test_case)


if __name__ == "__main__":
    unittest.main()
