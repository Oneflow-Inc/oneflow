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

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestLog1pModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_log1p_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return torch.log1p(x)

    @autotest(check_graph=True)
    def test_log1p_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        return torch.log1p(x)


if __name__ == "__main__":
    unittest.main()
