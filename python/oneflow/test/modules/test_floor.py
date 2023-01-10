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


def _test_floor(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.floor(of_input)
    np_out = np.floor(np_input)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = np.zeros_like(of_out, dtype=np.float32)
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_out_grad, 0.0001, 0.0001, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestFloor(flow.unittest.TestCase):
    def test_floor(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_floor(test_case, *arg)

    @autotest(check_graph=True)
    def test_flow_floor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.floor(x)
        return y

    @autotest(check_graph=True)
    def test_flow_floor_inplace_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x + 1
        y.floor_()
        return y

    @autotest(check_graph=True)
    def test_flow_floor_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.floor(x)
        return y

    @profile(torch.floor)
    def profile_floor(test_case):
        torch.floor(torch.ones(100, 100, 100))


if __name__ == "__main__":
    unittest.main()
