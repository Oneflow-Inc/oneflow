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
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestAtLeast(flow.unittest.TestCase):
    @autotest(n=5)
    def test_atleast_1d_with_list_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=2).to(device)
        out = torch.atleast_1d([x, y])
        return out

    @autotest(n=5)
    def test_atleast_1d_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(low=0, high=3).to(int)).to(device)
        out = torch.atleast_1d(x)
        return out

    @autotest(n=5)
    def test_atleast_2d_with_list_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=1).to(device)
        z = random_tensor(ndim=3).to(device)
        out = torch.atleast_2d([x, y, z])
        return out

    @autotest(n=5)
    def test_atleast_2d_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(low=0, high=4).to(int)).to(device)
        out = torch.atleast_2d(x)
        return out

    @autotest(n=5)
    def test_atleast_3d_with_list_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=1).to(device)
        z = random_tensor(ndim=2).to(device)
        p = random_tensor(ndim=4).to(device)
        out = torch.atleast_3d([x, y, z, p])
        return out

    @autotest(n=5)
    def test_atleast_3d_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(low=0, high=5).to(int)).to(device)
        out = torch.atleast_3d(x)
        return out


if __name__ == "__main__":
    unittest.main()
