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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_div_forward_backward(test_case, shape, device, dtype):
    x_arry = np.random.randn(*shape)
    y_arry = np.random.randn(*shape)
    # broadcast_div_grad of kCPU not support float16
    requires_grad = True if dtype is flow.float32 else False
    x = flow.tensor(x_arry, device=flow.device(device), dtype=dtype)
    y = flow.tensor(
        y_arry, device=flow.device(device), dtype=dtype, requires_grad=requires_grad
    )
    x_cpu = flow.tensor(x_arry, device=flow.device("cpu"), dtype=dtype)
    y_cpu = flow.tensor(
        y_arry, device=flow.device("cpu"), dtype=dtype, requires_grad=requires_grad
    )
    of_out = flow.div(x, y)
    cpu_out = flow.div(x_cpu, y_cpu)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), cpu_out.numpy(), 0.0001, 0.0001, equal_nan=True)
    )

    if requires_grad:
        s = of_out.sum()
        s_cpu = cpu_out.sum()
        s.backward()
        s_cpu.backward()
        test_case.assertTrue(
            np.allclose(y.grad.numpy(), y_cpu.grad.numpy(), 0.0001, 0.0001)
        )


def _test_broadcast_div_forward_backward(test_case, shapes, device, dtype):
    shape1 = shapes[0]
    shape2 = shapes[1]
    # broadcast_div_grad of kCPU not support float16
    requires_grad = True if dtype is flow.float32 else False
    x_arry = np.random.randn(*shape1)
    y_arry = np.random.randn(*shape2)
    x = flow.tensor(x_arry, device=flow.device(device), dtype=dtype)
    y = flow.tensor(
        y_arry, device=flow.device(device), dtype=dtype, requires_grad=requires_grad
    )
    x_cpu = flow.tensor(x_arry, device=flow.device("cpu"), dtype=dtype)
    y_cpu = flow.tensor(
        y_arry, device=flow.device("cpu"), dtype=dtype, requires_grad=requires_grad
    )
    of_out = flow.div(x, y)
    cpu_out = flow.div(x_cpu, y_cpu)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), cpu_out.numpy(), 0.0001, 0.0001, equal_nan=True)
    )

    if requires_grad:
        s = of_out.sum()
        s_cpu = cpu_out.sum()
        s.backward()
        s_cpu.backward()
        test_case.assertTrue(
            np.allclose(y.grad.numpy(), y_cpu.grad.numpy(), 0.0001, 0.0001)
        )


@flow.unittest.skip_unless_1n1d()
class TestDivCambriconModule(flow.unittest.TestCase):
    def test_div(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_div_forward_backward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.uint8,
            flow.int32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_broadcast_div(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_broadcast_div_forward_backward,
        ]
        arg_dict["shapes"] = list(
            zip(
                [
                    (2,),
                    (1, 3),
                    (2, 3, 4),
                    (2, 3, 4, 5),
                    (2, 1, 4, 5),
                    (2, 3, 4, 1),
                    (2, 3, 1, 5),
                    (2, 3, 4, 5),
                ],
                [(1,), (2, 1), (2, 1, 4), (2, 1, 1, 5), (4, 5), (1, 5), (4, 5), (1,)],
            )
        )
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.uint8,
            flow.int32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_0_size_div(test_case):
        x = flow.tensor(1.0, device=flow.device("mlu"), dtype=flow.float32)
        y = flow.tensor(2.0, device=flow.device("mlu"), dtype=flow.float32)
        z = x + y
        test_case.assertTrue(np.allclose(z.numpy(), [3.0], 0.0001, 0.0001))


if __name__ == "__main__":
    unittest.main()
