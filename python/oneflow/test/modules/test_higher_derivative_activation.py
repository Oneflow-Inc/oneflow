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


def print_value(name, value, debug=True):
    if not debug:
        return
    padl = (80 - len(name)) // 2
    padr = 80 - len(name) - padl
    print("-" * padl, name, "-" * padr)
    print(value)


def _test_activation_grad_grad_impl(test_case, op_name, *args, **kwargs):
    x = random_tensor(ndim=2, low=-1, dim1=4)
    y = eval(f"torch.nn.functional.{op_name}")(x, *args, **kwargs)

    x_shape = x.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape)
    init_grad_y = random_tensor(len(x_shape), *x_shape)
    init_grad_x = torch.ones(x_shape).requires_grad_()
    init_grad_y = torch.ones(x_shape).requires_grad_()

    print_value("x", x)
    print_value("y", y)
    print_value("init_grad_x", init_grad_x)
    print_value("init_grad_y", init_grad_y)

    dx = torch.autograd.grad(y, x, init_grad_y, True, True)[0]
    print_value("dx", dx)
    print(np.allclose(dx.pytorch.detach().cpu().numpy(), dx.oneflow.detach().numpy()))

    ddx_ddy = torch.autograd.grad(dx, [x, init_grad_y], init_grad_x)
    ddx, ddy = ddx_ddy[0], ddx_ddy[1]
    print_value("ddx", ddx)
    print_value("ddy", ddy)
    print(np.allclose(ddx.pytorch.detach().cpu().numpy(), ddx.oneflow.detach().numpy()))
    print(np.allclose(ddy.pytorch.detach().cpu().numpy(), ddy.oneflow.detach().numpy()))


def _test_prelu_activation_grad_grad_impl(test_case, op_name, *args, **kwargs):
    x = random_tensor(ndim=2, low=-1, dim1=4)
    a = random_tensor(ndim=1, dim0=x.oneflow.shape[1])
    y = torch.nn.functional.prelu(x, a)

    x_shape = x.oneflow.shape
    a_shape = a.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape)
    init_grad_y = random_tensor(len(x_shape), *x_shape)
    init_grad_a = random_tensor(len(a_shape), *a_shape)

    print_value("x", x)
    print_value("y", y)

    dx_and_da = torch.autograd.grad(y, [x, a], init_grad_y, True, True)
    dx, da = dx_and_da[0], dx_and_da[1]
    print_value("dx", dx)
    print_value("da", da)
    print(np.allclose(dx.pytorch.detach().cpu().numpy(), dx.oneflow.detach().numpy()))
    print(np.allclose(da.pytorch.detach().cpu().numpy(), da.oneflow.detach().numpy()))

    ddx_dda_ddy = torch.autograd.grad(
        dx_and_da, [dx, da, init_grad_y], [init_grad_x, init_grad_a]
    )
    ddx, dda, ddy = ddx_dda_ddy[0], ddx_dda_ddy[1], ddx_dda_ddy[2]
    print_value("ddx", ddx)
    print_value("dda", dda)
    print_value("ddy", ddy)
    print(np.allclose(ddx.pytorch.detach().cpu().numpy(), ddx.oneflow.detach().numpy()))
    print(np.allclose(dda.pytorch.detach().cpu().numpy(), dda.oneflow.detach().numpy()))
    print(np.allclose(ddy.pytorch.detach().cpu().numpy(), ddy.oneflow.detach().numpy()))


def _test_hardswish_activation_grad_grad_impl(test_case, op_name, *args, **kwargs):
    x = random_tensor(ndim=2, low=-1, dim1=4)
    y = torch.nn.functional.hardswish(x, *args, **kwargs)

    x_shape = x.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape)
    init_grad_y = random_tensor(len(x_shape), *x_shape)

    print_value("x", x)
    print_value("y", y)
    print_value("init_grad_x", init_grad_x)
    print_value("init_grad_y", init_grad_y)

    dx_pytorch = pytorch_origin.autograd.grad(
        y.pytorch, x.pytorch, init_grad_y.pytorch
    )[0]
    dx_oneflow = oneflow_origin.autograd.grad(
        y.oneflow, x.oneflow, init_grad_y.oneflow, True, True
    )[0]
    print_value("dx_oneflow", dx_oneflow)
    print_value("dx_pytorch", dx_pytorch)
    print(np.allclose(dx_pytorch.detach().cpu().numpy(), dx_oneflow.detach().numpy()))

    ddx, ddy = flow.autograd.grad(
        dx_oneflow, [x.oneflow, init_grad_y.oneflow], init_grad_x.oneflow
    )

    x, dx, init_grad_x, init_grad_y = (
        x.oneflow,
        dx_oneflow,
        init_grad_x.oneflow,
        init_grad_y.oneflow,
    )
    manual_ddx = flow.where(
        ((x > -3.0) < 3.0), 1.0 / 3.0 * init_grad_x * init_grad_y, flow.tensor(0.0)
    )
    manual_ddy = dx / init_grad_y * init_grad_x
    print_value("ddx", ddx)
    print_value("manual_ddx", manual_ddx)
    print_value("ddy", ddy)
    print(np.allclose(manual_ddx.detach().numpy(), ddx.detach().numpy()))
    print(np.allclose(manual_ddy.detach().numpy(), ddy.detach().numpy()))

def _test_hardsigmoid_activation_grad_grad_impl(test_case, op_name, *args, **kwargs):
    x = random_tensor(ndim=2, low=-1, dim1=4)
    y = torch.nn.functional.hardsigmoid(x, *args, **kwargs)

    x_shape = x.oneflow.shape
    init_grad_x = random_tensor(len(x_shape), *x_shape)
    init_grad_y = random_tensor(len(x_shape), *x_shape)

    print_value("x", x)
    print_value("y", y)
    print_value("init_grad_x", init_grad_x)
    print_value("init_grad_y", init_grad_y)

    dx_pytorch = pytorch_origin.autograd.grad(
        y.pytorch, x.pytorch, init_grad_y.pytorch
    )[0]
    dx_oneflow = oneflow_origin.autograd.grad(
        y.oneflow, x.oneflow, init_grad_y.oneflow, True, True
    )[0]
    print_value("dx_oneflow", dx_oneflow)
    print_value("dx_pytorch", dx_pytorch)
    print(np.allclose(dx_pytorch.detach().cpu().numpy(), dx_oneflow.detach().numpy()))

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
    print_value("ddx", ddx)
    print_value("manual_ddx", manual_ddx)
    print_value("ddy", ddy)
    print_value("manual_ddy", manual_ddy)
    print(np.allclose(manual_ddx.detach().numpy(), ddx.detach().numpy()))
    print(np.allclose(manual_ddy.detach().numpy(), ddy.detach().numpy()))


class TestActivationHigherDerivative(flow.unittest.TestCase):
    def test_activation_grad_grad(test_case):
        from collections import defaultdict

        op_args = defaultdict(list)
        op_kwargs = defaultdict(dict)

        op_args["leaky_relu"] = [random(-1, 1)]
        op_args["hardtanh"] = [random(-5, -1), random(1, 5)]
        op_args["elu"] = [random(-1, 1)]
        op_kwargs["celu"] = {"alpha": random(-1, 1)}
        op_kwargs["threshold"] = {"threshold": random(), "value": random()}

        # not done
        op_names = ["mish", "gelu"]

        # passed
        op_names = [
            "relu",
            "prelu",
            "hardtanh",
            "hardshrink",
            "softshrink",
            "selu",
            "elu",
            "celu",
            "softsign",
            "leaky_relu",
            "softplus",
            "hardswish","hardsigmoid"
        ]

        # totest
        # op_names = ["mish", "gelu", "silu"]
        op_names = ["silu"]

        for op_name in op_names:
            try:
                functor = eval(f"_test_{op_name}_activation_grad_grad_impl")
            except:
                functor = _test_activation_grad_grad_impl
            print(">" * 35, op_name, "<" * (40-len(op_name)))
            functor(test_case, op_name, *op_args[op_name], **op_kwargs[op_name])


if __name__ == "__main__":
    unittest.main()
