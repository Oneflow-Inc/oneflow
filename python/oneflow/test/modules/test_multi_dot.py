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

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


class TestMultiDotModule(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_multi_dot_random_1d_tensors(test_case):
        device = random_device()
        k = random(10, 100)
        x = random_tensor(ndim=1, dim0=k).to(device)
        y = random_tensor(ndim=1, dim0=k).to(device)
        return torch.linalg.multi_dot([x, y])

    @autotest(check_graph=False)
    def test_multi_dot_random_first_1d_tensor(test_case):
        device = random_device()
        k = random(10, 100)
        x = random_tensor(ndim=1, dim0=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        return torch.linalg.multi_dot([x, y])

    @autotest(check_graph=False)
    def test_multi_dot_random_last_1d_tensor(test_case):
        device = random_device()
        k = random(10, 100)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=1, dim0=k).to(device)
        return torch.linalg.multi_dot([x, y])

    @autotest(check_graph=False)
    def test_multi_dot_random_multi_2d_tensors(test_case):
        device = random_device()
        k0 = random(5, 20)
        k1 = random(5, 20)
        x = random_tensor(ndim=2, dim1=k0).to(device)
        y = random_tensor(ndim=2, dim0=k0, dim1=k1).to(device)
        z = random_tensor(ndim=2, dim0=k1).to(device)
        return torch.linalg.multi_dot([x, y, z])


if __name__ == "__main__":
    unittest.main()
