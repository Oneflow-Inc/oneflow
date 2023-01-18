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


def _test_unsqueeze(test_case, device):
    np_arr = np.random.rand(2, 6, 9, 3)
    x = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    y = flow.unsqueeze(x, dim=1)
    output = np.expand_dims(np_arr, axis=1)
    test_case.assertTrue(np.allclose(output, y.numpy(), 1e-05, 1e-05))
    x_flow = flow.randn(5)
    x_flow = flow.unsqueeze(x_flow, 0)
    test_case.assertTrue(np.array_equal(x_flow.stride(), (5, 1)))
    x_flow = flow.randn(5, 2)
    x_flow = flow.unsqueeze(x_flow, 0)
    test_case.assertTrue(np.array_equal(x_flow.stride(), (10, 2, 1)))


def _test_unsqueeze_tensor_function(test_case, device):
    np_arr = np.random.rand(2, 3, 4)
    x = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    y = x.unsqueeze(dim=2)
    output = np.expand_dims(np_arr, axis=2)
    test_case.assertTrue(np.allclose(output, y.numpy(), 1e-05, 1e-05))


def _test_unsqueeze_different_dim(test_case, device):
    np_arr = np.random.rand(4, 5, 6, 7)
    x = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    for axis in range(-5, 5):
        y = flow.unsqueeze(x, dim=axis)
        output = np.expand_dims(np_arr, axis=axis)
        test_case.assertTrue(np.allclose(output, y.numpy(), 1e-05, 1e-05))


def _test_unsqueeze_backward(test_case, device):
    np_arr = np.random.rand(2, 3, 4, 5)
    x = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = flow.unsqueeze(x, dim=1).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.ones((2, 3, 4, 5)), 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestUnsqueeze(flow.unittest.TestCase):
    def test_unsqueeze(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_unsqueeze,
            _test_unsqueeze_tensor_function,
            _test_unsqueeze_different_dim,
            _test_unsqueeze_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(check_graph=True)
    def test_flow_unsqueeze_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.unsqueeze(x, random(1, 3).to(int))
        return y

    @autotest(n=10, check_graph=False, auto_backward=False)
    def test_inplace_unsqueeze_with_random_data(test_case):
        device = random_device()
        x = random_tensor(requires_grad=False).to(device)
        y = x.unsqueeze_(random(1, 3).to(int))
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_unsqueeze_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(3, 2, 1, 0).to(device)
        y = torch.unsqueeze(x, random(0, 2).to(int))
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_unsqueeze_bool_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device=device, dtype=torch.bool)
        y = torch.unsqueeze(x, random(1, 3).to(int))
        return y


if __name__ == "__main__":
    unittest.main()
