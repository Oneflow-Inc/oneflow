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

import torch as pytorch_origin
import oneflow as oneflow_origin
from collections import defaultdict


def _assert_true(test_case, value1, value2):
    test_case.assertTrue(
        np.allclose(
            value1.detach().cpu().numpy(),
            value2.detach().numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
    )


def _test_activation_grad_grad_impl(test_case, op_name, placement, *args, **kwargs):
    x = random_tensor(ndim=2, low=-5, dim0=8, dim1=8).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    y = eval(f"torch.nn.functional.{op_name}")(x, *args, **kwargs)

    x_shape = x.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    init_grad_y = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    _assert_true(test_case, dx.pytorch, dx.oneflow)

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]
    _assert_true(test_case, ddx.pytorch, ddx.oneflow)
    _assert_true(test_case, ddy.pytorch, ddy.oneflow)


def _test_prelu_activation_grad_grad_impl(
    test_case, op_name, placement, *args, **kwargs
):
    x = random_tensor(ndim=2, low=-5, dim0=8, dim1=8).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    a = random_tensor(ndim=1, dim0=x.oneflow.shape[1]).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1)
    )
    y = torch.nn.functional.prelu(x, a)

    x_shape = x.oneflow.shape
    a_shape = a.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    init_grad_y = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    init_grad_a = random_tensor(len(a_shape), *a_shape).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1)
    )

    dx_and_da = torch.autograd.grad(y, [x, a], init_grad_y, True, True)
    dx, da = dx_and_da[0], dx_and_da[1]
    _assert_true(test_case, dx.pytorch, dx.oneflow)
    _assert_true(test_case, da.pytorch, da.oneflow)

    ddx_dda_ddy = torch.autograd.grad(
        dx_and_da, [dx, da, init_grad_y], [init_grad_x, init_grad_a], True, True
    )
    ddx, dda, ddy = ddx_dda_ddy[0], ddx_dda_ddy[1], ddx_dda_ddy[2]
    _assert_true(test_case, ddx.pytorch, ddx.oneflow)
    _assert_true(test_case, dda.pytorch, dda.oneflow)
    _assert_true(test_case, ddy.pytorch, ddy.oneflow)


def _test_hardswish_activation_grad_grad_impl(
    test_case, op_name, placement, *args, **kwargs
):
    x = random_tensor(ndim=2, low=-1, dim0=8, dim1=8).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    y = torch.nn.functional.hardswish(x, *args, **kwargs)

    x_shape = x.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    init_grad_y = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )

    dx_pytorch = pytorch_origin.autograd.grad(
        y.pytorch, x.pytorch, init_grad_y.pytorch
    )[0]
    dx_oneflow = oneflow_origin.autograd.grad(
        y.oneflow, x.oneflow, init_grad_y.oneflow, True, True
    )[0]
    _assert_true(test_case, dx_pytorch, dx_oneflow)

    ddx, ddy = flow.autograd.grad(
        dx_oneflow, [x.oneflow, init_grad_y.oneflow], init_grad_x.oneflow
    )
    x, dx, init_grad_x, init_grad_y = (
        x.oneflow,
        dx_oneflow,
        init_grad_x.oneflow,
        init_grad_y.oneflow,
    )

    zeros_grad = flow.zeros_like(x).to_global(placement=placement, sbp=x.sbp)
    manual_ddx = flow.where(
        ((x > -3.0) < 3.0), 1.0 / 3.0 * init_grad_x * init_grad_y, zeros_grad
    )
    manual_ddy = dx / init_grad_y * init_grad_x
    _assert_true(test_case, manual_ddx, ddx)
    _assert_true(test_case, manual_ddy, ddy)


def _test_hardsigmoid_activation_grad_grad_impl(
    test_case, op_name, placement, *args, **kwargs
):
    x = random_tensor(ndim=2, low=-1, dim0=8, dim1=8).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    y = torch.nn.functional.hardsigmoid(x, *args, **kwargs)

    x_shape = x.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )
    init_grad_y = random_tensor(len(x_shape), *x_shape).to_global(
        placement=placement, sbp=random_sbp(placement=placement, max_dim=2)
    )

    dx_pytorch = pytorch_origin.autograd.grad(
        y.pytorch, x.pytorch, init_grad_y.pytorch
    )[0]
    dx_oneflow = oneflow_origin.autograd.grad(
        y.oneflow, x.oneflow, init_grad_y.oneflow, True, True
    )[0]
    _assert_true(test_case, dx_pytorch, dx_oneflow)

    ddx, ddy = flow.autograd.grad(
        dx_oneflow, [x.oneflow, init_grad_y.oneflow], init_grad_x.oneflow
    )
    x, dx, init_grad_x, init_grad_y = (
        x.oneflow,
        dx_oneflow,
        init_grad_x.oneflow,
        init_grad_y.oneflow,
    )
    manual_ddx = flow.zeros_like(x)
    manual_ddy = dx / init_grad_y * init_grad_x
    _assert_true(test_case, manual_ddx, ddx)
    _assert_true(test_case, manual_ddy, ddy)


class TestActivationHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_activation_grad_grad(test_case):
        op_args = defaultdict(list)
        op_kwargs = defaultdict(dict)

        # parameter name not same in pytorch and oneflow
        op_args["leaky_relu"] = [random(-1, 1).to(float)]

        # some op only support kwargs, like celu in oneflow
        op_kwargs["hardtanh"] = {
            "min_val": random(-5, -1).to(float),
            "max_val": random(1, 5).to(float),
        }
        op_kwargs["elu"] = {"alpha": random(0, 10).to(float)}
        op_kwargs["celu"] = {"alpha": random(0, 10).to(float)}
        op_kwargs["threshold"] = {
            "threshold": random().to(float),
            "value": random().to(float),
        }
        op_kwargs["softplus"] = {
            "beta": random().to(float),
            "threshold": random().to(float),
        }

        op_names = [
            "mish",
            "gelu",
            "silu",
            "selu",
            "softsign",
            "hardsigmoid",
            "hardswish",
            "relu",
            "elu",
            "celu",
            "prelu",
            "hardshrink",
            "softshrink",
            "leaky_relu",
            "hardtanh",
            "softplus",
            "threshold",
        ]
        for op_name in op_names:
            try:
                functor = eval(f"_test_{op_name}_activation_grad_grad_impl")
            except:
                functor = _test_activation_grad_grad_impl

            print(f"| {op_name:-^60} |")
            for placement in all_placement():
                for i in range(1):
                    functor(
                        test_case,
                        op_name,
                        placement,
                        *op_args[op_name],
                        **op_kwargs[op_name],
                    )


if __name__ == "__main__":
    unittest.main()
