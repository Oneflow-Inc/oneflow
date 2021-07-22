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

import oneflow.experimental as flow
from test_util import GenArgList
from automated_test_util import *


def _test_greater_normal(test_case, device):
    input1 = flow.Tensor(
        np.array([1, 1, 4]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    input2 = flow.Tensor(
        np.array([1, 2, 3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.gt(input1, input2)
    np_out = np.greater(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_symbol(test_case, device):
    input1 = flow.Tensor(
        np.array([1, 1, 4]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    input2 = flow.Tensor(
        np.array([1, 2, 3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = input1 > input2
    np_out = np.greater(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_int_scalar(test_case, device):
    np_arr = np.random.randn(2, 3, 4, 5)
    input1 = flow.Tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    input2 = 1
    of_out = input1 > input2
    np_out = np.greater(np_arr, input2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_int_tensor_int_scalar(test_case, device):
    np_arr = np.random.randint(2, size=(2, 3, 4, 5))
    input1 = flow.Tensor(np_arr, dtype=flow.int, device=flow.device(device))
    input2 = 1
    of_out = input1 > input2
    np_out = np.greater(np_arr, input2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_float_scalar(test_case, device):
    np_arr = np.random.randn(3, 2, 5, 7)
    input1 = flow.Tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    input2 = 2.3
    of_out = input1 > input2
    np_out = np.greater(np_arr, input2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


@flow.unittest.skip_unless_1n1d()
class TestGreater(flow.unittest.TestCase):
    def test_greater(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_greater_normal,
            _test_greater_symbol,
            _test_greater_int_scalar,
            _test_greater_int_tensor_int_scalar,
            _test_greater_float_scalar,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(auto_backward=False)
    def test_greater_with_random_data(test_case):
        device = random_device()
        x1 = random_pytorch_tensor(ndim=4, dim0=2, dim1=3, dim2=4, dim3=5, requires_grad=False).to(device)
        x2 = random_pytorch_tensor(ndim=4, dim0=2, dim1=3, dim2=4, dim3=5, requires_grad=False).to(device)
        y1 = torch.gt(x1, x2)
        y2 = torch.gt(x1, random().to(int))
        y3 = torch.gt(x1, random().to(float))
        return y1, y2, y3

    @autotest(auto_backward=False)
    def test_tensor_greater_with_random_data(test_case):
        device = random_device()
        x1 = random_pytorch_tensor(ndim=4, dim0=2, dim1=3, dim2=4, dim3=5, requires_grad=False).to(device)
        x2 = random_pytorch_tensor(ndim=4, dim0=2, dim1=3, dim2=4, dim3=5, requires_grad=False).to(device)
        y1 = x1.gt(x2)
        y2 = x1 > x2
        y3 = x1.gt(random().to(int))
        y4 = x1.gt(random().to(float))
        return y1, y2, y3, y4


if __name__ == "__main__":
    unittest.main()
