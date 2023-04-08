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
import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def _test_different_dtype(test_case, device, shape):
    y1 = flow.ones(shape, dtype=flow.int32, device=flow.device(device))
    test_case.assertTrue(np.array_equal(np.ones(shape, dtype=np.int32), y1.numpy()))
    y2 = flow.ones(shape, dtype=flow.uint8, device=flow.device(device))
    test_case.assertTrue(np.array_equal(np.ones(shape, dtype=np.uint8), y2.numpy()))
    y3 = flow.ones(shape, dtype=flow.float64, device=flow.device(device))
    test_case.assertTrue(np.array_equal(np.ones(shape, dtype=np.float64), y3.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestConstantModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_flow_zeros_list_with_random_data(test_case):
        device = random_device()
        y1 = torch.zeros(random().to(int)).to(device)
        y2 = torch.zeros(random().to(int), random().to(int)).to(device)
        y3 = torch.zeros(random().to(int), random().to(int), random().to(int)).to(
            device
        )
        y4 = torch.zeros(
            random().to(int), random().to(int), random().to(int), random().to(int)
        ).to(device)
        return y1, y2, y3, y4

    @profile(torch.zeros)
    def profile_zeros(test_case):
        torch.zeros(2, 3)
        torch.zeros(32, 3, 128, 128)
        torch.zeros(1000, 1000)

    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_flow_ones_list_with_random_data(test_case):
        device = random_device()
        y1 = torch.ones(random().to(int)).to(device)
        y2 = torch.ones(random().to(int), random().to(int)).to(device)
        y3 = torch.ones(random().to(int), random().to(int), random().to(int)).to(device)
        y4 = torch.ones(
            random().to(int), random().to(int), random().to(int), random().to(int)
        ).to(device)
        return y1, y2, y3, y4

    @profile(torch.ones)
    def profile_ones(test_case):
        torch.ones(2, 3)
        torch.ones(32, 3, 128, 128)
        torch.ones(1000, 1000)

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_zeros_like_list_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.zeros_like(x)
        return y

    @profile(torch.zeros_like)
    def profile_zeros_like(test_case):
        input1 = torch.ones(32, 3, 128, 128)
        input2 = torch.ones(1000, 1000)
        input3 = torch.ones(2, 3)
        torch.zeros_like(input1)
        torch.zeros_like(input2)
        torch.zeros_like(input3)

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_zeros_like_list_with_random_data_and_requires_grad(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.zeros_like(x, requires_grad=True)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_zeros_like_list_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.zeros_like(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_ones_like_list_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.ones_like(x)
        return y

    @profile(torch.ones_like)
    def profile_ones_like(test_case):
        input1 = torch.ones(32, 3, 128, 128)
        input2 = torch.ones(1000, 1000)
        input3 = torch.ones(2, 3)
        torch.ones_like(input1)
        torch.ones_like(input2)
        torch.ones_like(input3)

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_ones_like_list_with_random_data_and_requires_grad(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.ones_like(x, requires_grad=True)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_ones_like_list_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.ones_like(x)
        return y

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_new_ones_list_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.new_ones(
            (random().to(int), random().to(int), random().to(int)),
            device=device.value(),
            requires_grad=constant(True),
        )
        return y

    @profile(torch.Tensor.new_ones)
    def profile_new_ones(test_case):
        x = torch.Tensor(np.ones((1, 2, 3)))
        x.new_ones((2, 3))
        x.new_ones((32, 3, 128, 128))
        x.new_ones((1000, 1000, 1000, 1000))

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_new_ones_list_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = x.new_ones(
            (random().to(int), random().to(int), random().to(int)),
            device=device.value(),
            requires_grad=constant(True),
        )
        return y

    @autotest(n=5)
    def test_new_zeros(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.new_zeros(
            (random().to(int), random().to(int), random().to(int)),
            device=device.value(),
            requires_grad=constant(True),
        )
        return y

    @profile(torch.Tensor.new_zeros)
    def profile_new_zeros(test_case):
        x = torch.Tensor(np.ones((1, 2, 3)))
        x.new_zeros((2, 3))
        x.new_zeros((32, 3, 128, 128))
        x.new_zeros((1000, 1000, 1000, 1000))

    @autotest(n=5)
    def test_new_full(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.new_full(
            (random().to(int), random().to(int), random().to(int)),
            random().to(float).value(),
            device=device.value(),
            requires_grad=constant(True),
        )
        return y

    @autotest(n=5, auto_backward=False)
    def test_new_full_with_scalar(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.new_full([], random().to(int))
        return y

    @autotest(n=5, auto_backward=False)
    def test_full_with_scalar(test_case):
        device = random_device()
        y = torch.full([], random().to(int), device=device)
        return y

    @autotest(n=10, auto_backward=True)
    def test_full_with_random_data_int(test_case):
        device = random_device()
        shape = random_tensor(low=1, high=6, requires_grad=False).pytorch.shape
        y = torch.full(shape, 2.0, requires_grad=True)
        return y

    @autotest(n=5)
    def test_full_with_random_data_numpy_scalar(test_case):
        device = random_device()
        shape = random_tensor(low=1, high=6, requires_grad=False).pytorch.shape
        y = torch.full(shape, np.array([2.0])[0], device=device, requires_grad=True)
        return y

    @autotest(n=5)
    def test_full_with_scalar_tensor(test_case):
        device = random_device()
        shape = random_tensor(low=0, high=6, requires_grad=False).pytorch.shape
        y = torch.full(
            shape,
            torch.tensor(2.0, requires_grad=random().to(bool)),
            device=device,
            requires_grad=True,
        )
        return y

    @profile(torch.full)
    def profile_full_with_scalar_tensor(test_case):
        torch.full((2, 3), torch.tensor(3.141592))
        torch.full((64, 3, 128, 128), torch.tensor(3.141592))
        torch.full((1000, 1000), torch.tensor(3.141592))

    @profile(torch.full)
    def profile_full(test_case):
        torch.full((2, 3), 3.141592)
        torch.full((64, 3, 128, 128), 3.141592)
        torch.full((1000, 1000), 3.141592)

    @autotest(n=10, auto_backward=True)
    def test_full_with_random_data_float(test_case):
        device = random_device()
        shape = random_tensor(low=1, high=6, requires_grad=False).pytorch.shape
        y = torch.full(shape, 2.0, requires_grad=True)
        return y

    @autotest(n=10, auto_backward=True)
    def test_full_like_with_random_data_float(test_case):
        device = random_device()
        x = random_tensor(low=1, high=6, requires_grad=False).to(device)
        y = torch.full_like(x, 2.0, requires_grad=True)
        return y

    @profile(torch.full_like)
    def profile_full_like(test_case):
        torch.full_like(torch.ones(2, 3), 3.141592)
        torch.full_like(torch.ones(64, 3, 128, 128), 3.141592)
        torch.full_like(torch.ones(1000, 1000), 3.141592)

    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_different_dtype,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 0, 4)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
