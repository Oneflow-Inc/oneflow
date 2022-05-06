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

import copy
import os
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    @autotest(n=10, auto_backward=True, check_graph=False)
    def test_tensor_pin_memory(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        x2 = x.pin_memory()
        x3 = x2.pin_memory()
        test_case.assertTrue(id(x.pytorch) != id(x2.pytorch))
        test_case.assertTrue(id(x3.pytorch) == id(x2.pytorch))
        test_case.assertTrue(id(x.oneflow) != id(x2.oneflow))
        test_case.assertTrue(id(x3.oneflow) == id(x2.oneflow))
        return x3
    
    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_0_dim_tensor_pin_memory(test_case):
        device = random_device()
        x = random_tensor(ndim=1).to(device)
        x1 = x[0]
        x2 = x1.pin_memory()
        x3 = x2.pin_memory()
        test_case.assertTrue(id(x1.pytorch) != id(x2.pytorch))
        test_case.assertTrue(id(x3.pytorch) == id(x2.pytorch))
        test_case.assertTrue(id(x1.oneflow) != id(x2.oneflow))
        test_case.assertTrue(id(x3.oneflow) == id(x2.oneflow))
        return x3


if __name__ == "__main__":
    unittest.main()
