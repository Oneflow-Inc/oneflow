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


def _test_squeeze(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    of_shape = flow.squeeze(input, dim=[1, 2]).numpy().shape
    np_shape = (1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(
        np.allclose(
            flow.squeeze(input, dim=[1, 2]).numpy(),
            np.squeeze(input.numpy(), axis=(1, 2)),
            0.0001,
            0.0001,
        )
    )


def _test_squeeze_1d_input(test_case, device):
    np_arr = np.random.rand(10)
    input = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    output = flow.squeeze(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_arr, 1e-05, 1e-05))


def _test_tensor_squeeze(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    of_shape = input.squeeze(dim=[1, 2]).numpy().shape
    np_shape = (1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(
        np.allclose(
            input.squeeze(dim=[1, 2]).numpy(),
            np.squeeze(input.numpy(), axis=(1, 2)),
            0.0001,
            0.0001,
        )
    )


def _test_squeeze_int(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    of_shape = flow.squeeze(input, 1).numpy().shape
    np_shape = (1, 1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(
        np.allclose(
            input.squeeze(1).numpy(), np.squeeze(input.numpy(), axis=1), 0.0001, 0.0001
        )
    )


def _test_squeeze_backward(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = flow.squeeze(input, dim=1).sum()
    y.backward()
    np_grad = np.ones((1, 1, 1, 3))
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


@flow.unittest.skip_unless_1n1d()
class TestSqueeze(flow.unittest.TestCase):
    def test_squeeze(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_squeeze,
            _test_squeeze_1d_input,
            _test_squeeze_int,
            _test_tensor_squeeze,
            _test_squeeze_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(check_graph=True)
    def test_flow_squeeze_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.squeeze(x, random(1, 3).to(int))
        return y

    @autotest(n=10, check_graph=False, auto_backward=False)
    def test_inplace_squeeze_with_random_data(test_case):
        device = random_device()
        x = random_tensor(requires_grad=False).to(device)
        y = x.squeeze_(random(1, 3).to(int))
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_squeeze_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(3, 2, 1, 0).to(device)
        y = torch.squeeze(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_squeeze_bool_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device=device, dtype=torch.bool)
        y = torch.squeeze(x, random(1, 3).to(int))
        return y


if __name__ == "__main__":
    unittest.main()
