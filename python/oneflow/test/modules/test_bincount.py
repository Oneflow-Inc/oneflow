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

import oneflow as flow
from oneflow.test_utils.automated_test_util import *
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestBinCount(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_bincount(test_case):
        device = random_device()
        x = random_tensor(1, 10000, low=0, high=65536, dtype=int).to(device)
        result = torch.bincount(x)
        return result

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_bincount_weight(test_case):
        device = random_device()
        x = random_tensor(1, 10000, low=0, high=65536, dtype=int).to(device)
        weight = random_tensor(1, 10000).to(device)
        return torch.bincount(x, weights=weight)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_bincount_minlength(test_case):
        device = random_device()
        x = random_tensor(1, 10000, low=0, high=65536, dtype=int).to(device)
        weight = random_tensor(1, 10000).to(device)
        minlength = random(1, 200).to(int)
        return torch.bincount(x, weights=weight, minlength=minlength)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_bincount_0element(test_case):
        device = random_device()
        x = random_tensor(1, 0, low=0, high=65536, dtype=int).to(device)
        weight = random_tensor(1, 0).to(device)
        minlength = random(1, 200).to(int)
        return torch.bincount(x, weights=weight, minlength=minlength)

    @profile(torch.bincount)
    def profile_bincount(test_case):
        torch.bincount(torch.ones(4096).int())
        torch.bincount(torch.ones(65536).int())
        torch.bincount(torch.arange(4096).int())


if __name__ == "__main__":
    unittest.main()
