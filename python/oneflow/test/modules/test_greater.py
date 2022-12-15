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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_greater_normal(test_case, device):
    input1 = flow.tensor(
        np.array([1, 1, 4]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    input2 = flow.tensor(
        np.array([1, 2, 3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.gt(input1, input2)
    np_out = np.greater(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_symbol(test_case, device):
    input1 = flow.tensor(
        np.array([1, 1, 4]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    input2 = flow.tensor(
        np.array([1, 2, 3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = input1 > input2
    np_out = np.greater(input1.numpy(), input2.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_int_scalar(test_case, device):
    np_arr = np.random.randn(2, 3, 4, 5)
    input1 = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    input2 = 1
    of_out = input1 > input2
    np_out = np.greater(np_arr, input2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_int_tensor_int_scalar(test_case, device):
    np_arr = np.random.randint(2, size=(2, 3, 4, 5))
    input1 = flow.tensor(np_arr, dtype=flow.int, device=flow.device(device))
    input2 = 1
    of_out = input1 > input2
    np_out = np.greater(np_arr, input2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_greater_float_scalar(test_case, device):
    np_arr = np.random.randn(3, 2, 5, 7)
    input1 = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
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

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_greater_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = torch.gt(x1, oneof(x2, random().to(int), random().to(float)))
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_tensor_inplace_greater_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x1.gt_(oneof(x2, random().to(int), random().to(float)))
        return x1

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_tensor_greater_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y1 = x1.gt(oneof(x2, random().to(int), random().to(float)))
        y2 = x1 > x2
        return (y1, y2)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_greater_with_0_size_data(test_case):
        device = random_device()
        x1 = random_tensor(4, 2, 3, 0, 5).to(device)
        x2 = random_tensor(4, 2, 3, 0, 5).to(device)
        y1 = torch.gt(x1, x2)
        y2 = x1 > x2
        return (y1, y2)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_greater_bool_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        y = torch.gt(x1, oneof(x2, random().to(int), random().to(float)))
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_greater_with_0dim_data(test_case):
        device = random_device()
        x1 = random_tensor(ndim=0).to(device)
        x2 = random_tensor(ndim=0).to(device)
        y1 = torch.gt(x1, x2)
        y2 = x1 > x2
        return (y1, y2)

    @profile(torch.gt)
    def profile_gt(test_case):
        input = torch.ones(1000, 1000)
        other = torch.ones(1000, 1000)
        torch.gt(input, other)
        torch.gt(input, 0)


if __name__ == "__main__":
    unittest.main()
