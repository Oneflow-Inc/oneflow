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


@flow.unittest.skip_unless_1n1d()
class TestUnbind(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True)
    def test_unbind_flow_with_random_data1(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.unbind(x, random(0, 4).to(int))
        return y

    @autotest(n=5, check_graph=True)
    def test_unbind_flow_with_random_data2(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.unbind(x, random(0, 4).to(int))
        return y

    @autotest(n=5, check_graph=True)
    def test_unbind_flow_with_random_data3(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        y = torch.unbind(x, random(0, 3).to(int))
        return y

    @autotest(n=5, check_graph=True)
    def test_unbind_flow_with_random_data4(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        y = torch.unbind(x, random(0, 3).to(int))
        return y

    @autotest(n=5, check_graph=True)
    def test_unbind_flow_with_random_data5(test_case):
        device = random_device()
        x = random_tensor(ndim=2).to(device)
        y = torch.unbind(x, random(0, 2).to(int))
        return y


if __name__ == "__main__":
    unittest.main()
