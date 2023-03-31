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

from oneflow.test_utils.automated_test_util import *


def _test_logical_and(test_case, shape, dtype, device):
    np_input = np.random.randint(3, size=shape)
    np_other = np.random.randint(3, size=shape)
    input = flow.tensor(np_input, dtype=dtype, device=flow.device(device))
    other = flow.tensor(np_other, dtype=dtype, device=flow.device(device))
    of_out = flow.logical_and(input, other)
    np_out = np.logical_and(np_input, np_other)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    x = torch.ones(3).byte()
    y = torch.ones(3).byte()

    z = (x & ~y).bool()
    test_case.assertTrue(np.array_equal(z.numpy(), [False, False, False]))


def _test_tensor_logical_and(test_case, shape, dtype, device):
    np_input = np.random.randint(3, size=shape)
    np_other = np.random.randint(3, size=shape)
    input = flow.tensor(np_input, dtype=dtype, device=flow.device(device))
    other = flow.tensor(np_other, dtype=dtype, device=flow.device(device))
    of_out = input.logical_and(other)
    np_out = np.logical_and(np_input, np_other)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tensor_scalar_logical_and(test_case, shape, scalar, dtype, device):
    np_input = np.random.randint(3, size=shape)
    input = flow.tensor(np_input, dtype=dtype, device=flow.device(device))
    of_out = input.logical_and(scalar)
    np_out = np.logical_and(np_input, scalar)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


@flow.unittest.skip_unless_1n1d()
class TestLogicalAndModule(flow.unittest.TestCase):
    def test_logical_and(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_logical_and,
            _test_tensor_logical_and,
        ]
        arg_dict["shape"] = [(2, 3), (2, 4, 5)]
        arg_dict["dtype"] = [flow.float32, flow.int32]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_scalar_logical_and(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_tensor_scalar_logical_and]
        arg_dict["shape"] = [(2, 3), (2, 4, 5)]
        arg_dict["scalar"] = [1, 0]
        arg_dict["dtype"] = [flow.float32, flow.int32]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_logical_and_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = torch.logical_and(x1, x2)
        return y

    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_logical_and_bool_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        y = torch.logical_and(x1, x2)
        return y


if __name__ == "__main__":
    unittest.main()
