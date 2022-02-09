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
class Test_Diag_module(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_diag_one_dim(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random()).to(device)
        return torch.diag(x)

    @autotest(check_graph=True)
    def test_diag_other_dim(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(), dim1=random()).to(device)
        return torch.diag(x)

    @autotest(auto_backward=False, check_graph=True)
    def test_diag_one_dim(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random()).to(device, torch.bool)
        return torch.diag(x)


if __name__ == "__main__":
    unittest.main()
