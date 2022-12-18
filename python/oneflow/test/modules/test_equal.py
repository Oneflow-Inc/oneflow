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


@flow.unittest.skip_unless_1n1d()
class TestEqual(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_eq_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(3, 2, 0, 3).to(device)
        y = random_tensor(3, 2, 0, 3).to(device)
        z = torch.equal(x, y)
        return z

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_equal_with_0shape_0d_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        z = torch.equal(x, y)
        return z

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_flow_equal_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        return torch.equal(x, y)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_flow_tensor_equal_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        return x.equal(y)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_flow_equal_with_random_0d_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(ndim=0, requires_grad=False).to(device)
        y = random_tensor(ndim=0, requires_grad=False).to(device)
        return torch.equal(x, y)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_flow_equal_with_same_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        return torch.equal(x, x)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_flow_equal_bool_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        y = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        return torch.equal(x, y)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_flow_equal_with_same_random_0d_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(ndim=0, requires_grad=False).to(device)
        return torch.equal(x, x)

    @profile(torch.equal)
    def profile_equal(test_case):
        input1 = torch.ones(1000, 1280)
        input2 = torch.ones(1000, 1280)
        torch.equal(input1, input2)


if __name__ == "__main__":
    unittest.main()
