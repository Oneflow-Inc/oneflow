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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestCeilModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_ceil_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.ceil(input)
        return y

    @autotest(n=5)
    def test_ceil_flow_with_random_0d_data(test_case):
        device = random_device()
        input = random_tensor(ndim=0).to(device)
        y = torch.ceil(input)
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_ceil_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.ceil(x)
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_ceil_with_0shape_0d_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.ceil(x)
        return y

    @profile(torch.ceil)
    def profile_ceil(test_case):
        torch.ceil(torch.ones(4))
        torch.ceil(torch.ones(100000))


if __name__ == "__main__":
    unittest.main()
