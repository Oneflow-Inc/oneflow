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
class TestAbsModule(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True)
    def test_abs_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.abs(x)
        return y

    @autotest(n=5, check_graph=True)
    def test_abs_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.abs(x)
        return y

    @profile(torch.abs)
    def profile_abs(test_case):
        torch.abs(torch.ones(1, 128, 28, 28))
        torch.abs(torch.ones(16, 128, 28, 28))


if __name__ == "__main__":
    unittest.main()
