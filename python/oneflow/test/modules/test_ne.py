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

from oneflow.test_utils.automated_test_util import *


def _test_ne(test_case, shape, device):
    arr1 = np.random.randn(*shape)
    arr2 = np.random.randn(*shape)
    input = flow.tensor(arr1, dtype=flow.float32, device=flow.device(device))
    other = flow.tensor(arr2, dtype=flow.float32, device=flow.device(device))
    of_out = flow.ne(input, other)
    of_out2 = flow.not_equal(input, other)
    np_out = np.not_equal(arr1, arr2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    test_case.assertTrue(np.array_equal(of_out2.numpy(), np_out))
    test_case.assertTrue(input != None)
    test_case.assertTrue(None != input)


def _test_tensor_ne_operator(test_case, shape, device):
    arr1 = np.random.randn(*shape)
    arr2 = np.random.randn(*shape)
    input = flow.tensor(arr1, dtype=flow.float32, device=flow.device(device))
    other = flow.tensor(arr2, dtype=flow.float32, device=flow.device(device))
    of_out = input.ne(other)
    np_out = np.not_equal(arr1, arr2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_ne_int(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1
    of_out = flow.ne(input, num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tensor_ne_operator_int(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1
    of_out = input.ne(num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_ne_float(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1.0
    of_out = flow.ne(input, num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tensor_ne_operator_float(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1.0
    of_out = input.ne(num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


@flow.unittest.skip_unless_1n1d()
class TestNe(flow.unittest.TestCase):
    def test_ne(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_func"] = [
            _test_ne,
            _test_tensor_ne_operator,
            _test_ne_int,
            _test_tensor_ne_operator_int,
            _test_ne_float,
            _test_tensor_ne_operator_float,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_ne_with_0_size_data(test_case):
        device = random_device()
        x1 = random_tensor(4, 2, 3, 0, 5).to(device)
        x2 = random_tensor(4, 2, 3, 0, 5).to(device)
        y1 = torch.ne(x1, x2)
        y2 = torch.ne(x1, 2)
        y3 = torch.ne(x1, 2.0)
        return (y1, y2, y3)

    @autotest(n=5, auto_backward=False)
    def test_ne_with_0dim_data(test_case):
        device = random_device()
        x1 = random_tensor(ndim=0).to(device)
        x2 = random_tensor(ndim=0).to(device)
        y1 = torch.ne(x1, x2)
        y2 = torch.ne(x1, 2)
        y3 = torch.ne(x1, 2.0)
        return (y1, y2, y3)


if __name__ == "__main__":
    unittest.main()
