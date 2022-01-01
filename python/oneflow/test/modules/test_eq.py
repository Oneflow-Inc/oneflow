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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestEq(flow.unittest.TestCase):
    @autotest(auto_backward=False, check_graph=True)
    def test_eq_with_0_size_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(3, 2, 0, 3).to(device)
        y = random_pytorch_tensor(3, 2, 0, 3).to(device)
        z = torch.eq(x, y)
        return z

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_eq_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().value().shape
        x = random_pytorch_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = random_pytorch_tensor(len(shape), *shape, requires_grad=False).to(device)
        return torch.eq(x, y)

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_eq_with_same_random_data(test_case):
        device = random_device()
        shape = random_tensor().value().shape
        x = random_pytorch_tensor(len(shape), *shape, requires_grad=False).to(device)
        return torch.eq(x, x)


if __name__ == "__main__":
    unittest.main()
