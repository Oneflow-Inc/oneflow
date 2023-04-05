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


def _get_data(shape, dtype):
    array = np.random.randn(*shape)
    y = np.random.randn()
    if dtype == flow.int:
        array = array * 100
        array = array.astype(int)
        y = int(y * 100)
    return array, y


def _get_diff(dtype):
    return 0.001 if dtype == flow.float16 else 0.0001


def _test_scalar_add_forward(test_case, shape, device, dtype):
    array, y = _get_data(shape, dtype)
    x = flow.tensor(array, device=flow.device(device), dtype=dtype)
    of_out = x + y
    np_out = array + y
    diff = _get_diff(dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, diff, diff))


def _test_scalar_mul_forward(test_case, shape, device, dtype):
    array, y = _get_data(shape, dtype)
    x = flow.tensor(array, device=flow.device(device), dtype=dtype)
    of_out = x * y
    np_out = array * y
    diff = _get_diff(dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, diff, diff))


def _test_scalar_sub_forward(test_case, shape, device, dtype):
    array, y = _get_data(shape, dtype)
    x = flow.tensor(array, device=flow.device(device), dtype=dtype)
    of_out = x - y
    np_out = array - y
    diff = _get_diff(dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, diff, diff))


def _test_scalar_pow_forward(test_case, shape, device, dtype):
    if dtype == flow.int:
        return
    array, y = _get_data(shape, dtype)
    x = flow.tensor(array, device=flow.device(device), dtype=dtype)
    x_cpu = x.cpu()
    mlu_out = flow.pow(x, y)
    cpu_out = flow.pow(x_cpu, y)
    test_case.assertTrue(
        np.allclose(cpu_out.numpy(), mlu_out.numpy(), 0.0001, 0.0001, equal_nan=True)
    )


def _test_scalar_pow_backward(test_case, shape, device, dtype):
    if dtype == flow.int:
        return
    array, y = _get_data(shape, dtype)
    x = flow.tensor(array, device=flow.device(device), dtype=dtype, requires_grad=True)
    x_cpu = flow.tensor(
        array, device=flow.device("cpu"), dtype=dtype, requires_grad=True
    )
    mlu_out = flow.pow(x, y)
    cpu_out = flow.pow(x_cpu, y)
    mlu_out = mlu_out.sum()
    cpu_out = cpu_out.sum()
    mlu_out.backward()
    cpu_out.backward()

    # TODO(): The MLU precision error is usually relatively large when the
    # result value is greater than 1e7
    x_cpu_grad = x_cpu.grad.numpy()
    x_cpu_grad[np.abs(x_cpu_grad) > 1e7] = 1e7
    x_grad = x.grad.numpy()
    x_grad[np.abs(x_grad) > 1e7] = 1e7
    test_case.assertTrue(np.allclose(x_cpu_grad, x_grad, 0.01, 0.01, equal_nan=True))


@flow.unittest.skip_unless_1n1d()
class TestScalarMathCambriconModule(flow.unittest.TestCase):
    def test_scalar_math(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_scalar_add_forward,
            _test_scalar_mul_forward,
            _test_scalar_sub_forward,
            _test_scalar_pow_forward,
            _test_scalar_pow_backward,
        ]
        arg_dict["shape"] = [(4,), (4, 8), (2, 4, 8), (2, 4, 8, 2)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [flow.float, flow.float16, flow.int]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
