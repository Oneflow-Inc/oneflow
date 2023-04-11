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
class TestBaddBmmModule(flow.unittest.TestCase):
    @autotest(n=5, rtol=1e-4, atol=1e-3)
    def test_baddbmm_with_torch(test_case):
        device = random_device()
        input = random_tensor(ndim=3, dim0=2, dim1=4, dim2=4).to(device)
        batch1 = random_tensor(ndim=3, dim0=2, dim1=4, dim2=3).to(device)
        batch2 = random_tensor(ndim=3, dim0=2, dim1=3, dim2=4).to(device)
        y = torch.baddbmm(input, batch1, batch2, beta=2.0, alpha=1.2)
        return y

    @autotest(n=5, rtol=1e-4, atol=1e-3)
    def test_baddbmm_in_sd2_with_torch(test_case):
        device = random_device()
        input = random_tensor(ndim=3, dim0=2, dim1=2, dim2=2, requires_grad=False).to(
            device
        )
        batch1 = random_tensor(ndim=3, dim0=2, dim1=2, dim2=2).to(device)
        batch2 = random_tensor(ndim=3, dim0=2, dim1=2, dim2=2).to(device)
        y = torch.baddbmm(input, batch1, batch2, beta=0.0, alpha=1.2)
        return y

    @autotest(n=5, rtol=1e-4, atol=1e-3)
    def test_baddbmm_no_attr_with_torch(test_case):
        device = random_device()
        input = random_tensor(ndim=3, dim0=2, dim1=4, dim2=4).to(device)
        batch1 = random_tensor(ndim=3, dim0=2, dim1=4, dim2=3).to(device)
        batch2 = random_tensor(ndim=3, dim0=2, dim1=3, dim2=4).to(device)
        y = torch.baddbmm(input, batch1, batch2)
        return y

    @autotest(n=5, rtol=1e-4, atol=1e-3)
    def test_baddbmm_broadcast_with_torch(test_case):
        device = random_device()
        input = random_tensor(ndim=1, dim0=4).to(device)
        batch1 = random_tensor(ndim=3, dim0=2, dim1=4, dim2=3).to(device)
        batch2 = random_tensor(ndim=3, dim0=2, dim1=3, dim2=4).to(device)
        y = torch.baddbmm(input, batch1, batch2, beta=-1.98, alpha=1.34)
        return y

    @profile(torch.baddbmm)
    def profile_baddbmm(test_case):
        input = torch.ones(10, 100, 100)
        batch1 = torch.ones(10, 100, 100)
        batch2 = torch.ones(10, 100, 100)
        torch.bmm(input, batch1, batch2, beta=-1.98, alpha=1.34)


if __name__ == "__main__":
    unittest.main()
