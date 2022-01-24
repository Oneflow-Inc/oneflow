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
from test_util import GenArgList

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
    def test_consistent_naive(test_case):
        placement = flow.placement("cpu", {0: [0]})
        sbp = (flow.sbp.broadcast,)
        x = flow.ones((16, 16), placement=placement, sbp=sbp)
        test_case.assertEqual(x.sbp, sbp)
        test_case.assertEqual(x.placement, placement)

    @autotest(auto_backward=False, check_graph=True)
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

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_ones_list_with_random_data(test_case):
        device = random_device()
        y1 = torch.ones(random().to(int)).to(device)
        y2 = torch.ones(random().to(int), random().to(int)).to(device)
        y3 = torch.ones(random().to(int), random().to(int), random().to(int)).to(device)
        y4 = torch.ones(
            random().to(int), random().to(int), random().to(int), random().to(int)
        ).to(device)
        return y1, y2, y3, y4

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_zeros_like_list_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.zeros_like(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_zeros_like_list_with_0dim_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=0).to(device)
        y = torch.zeros_like(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_ones_like_list_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.ones_like(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_ones_like_list_with_0dim_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=0).to(device)
        y = torch.ones_like(x)
        return y

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_new_ones_list_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.new_ones(
            (random().to(int), random().to(int), random().to(int)),
            device=device.value(),
            requires_grad=constant(True),
        )
        return y

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_new_ones_list_with_0dim_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=0).to(device)
        y = x.new_ones(
            (random().to(int), random().to(int), random().to(int)),
            device=device.value(),
            requires_grad=constant(True),
        )
        return y

    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_different_dtype,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 0, 4)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

        @autotest(n=10, auto_backward=True)
        def test_full_with_random_data_int(test_case):
            device = random_device()
            shape = random_tensor().value().shape
            y = torch.full(shape, 2)
            return y

        @autotest(n=10, auto_backward=True)
        def test_full_with_random_data_float(test_case):
            device = random_device()
            shape = random_tensor().value().shape
            y = torch.full(shape, 2.0)
            return y


if __name__ == "__main__":
    unittest.main()
