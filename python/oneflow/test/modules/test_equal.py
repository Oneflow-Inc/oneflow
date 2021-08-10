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
from test_util import GenArgList
from automated_test_util import *

# TODO: equal的底层实现没有和pytorch对齐
@flow.unittest.skip_unless_1n1d()
class TestEqual(flow.unittest.TestCase):
    @autotest()
    def test_flow_equal_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().value().shape
        x = random_tensor(len(shape), *shape).to(device)
        y = random_tensor(len(shape), *shape).to(device)
        return torch.equal(x, y)

    @autotest()
    def test_flow_equal_with_same_random_data(test_case):
        device = random_device()
        shape = random_tensor().value().shape
        x = random_tensor(len(shape), *shape).to(device)
        return torch.equal(x, x)


if __name__ == "__main__":
    unittest.main()
