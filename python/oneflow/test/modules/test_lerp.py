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
class TestLerp(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_lerp_with_broadcast_data(test_case):
        device = random_device()
        start = random_tensor(ndim=2, dim0=3, dim1=1).to(device)
        end = random_tensor(ndim=2, dim0=1, dim1=3).to(device)
        weight = random_tensor(ndim=1, dim0=1).to(device)
        return torch.lerp(start, end, weight)

    @autotest()
    def test_lerp_with_random_data(test_case):
        device = random_device()
        start = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device)
        end = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device)
        weight = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device)
        return torch.lerp(
            start, end, oneof(weight, random().to(int), random().to(float))
        )

    @autotest()
    def test_tesnor_lerp_with_random_data(test_case):
        device = random_device()
        start = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device)
        end = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device)
        weight = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device)
        return start.lerp(end, oneof(weight, random().to(int), random().to(float)))

    @autotest()
    def test_tesnor_inplace_lerp_with_random_data(test_case):
        device = random_device()
        start = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device) + 0.01
        end = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device) + 0.01
        weight = random_tensor(ndim=3, dim0=3, dim1=4, dim2=5).to(device) + 0.01
        return start.lerp_(end, oneof(weight, random().to(int), random().to(float)))

    @profile(torch.lerp)
    def profile_lerp(test_case):
        torch.lerp(
            torch.randn(1, 32, 4, 4), torch.randn(1, 32, 4, 4), torch.randn(1, 32, 4, 4)
        )
        torch.lerp(
            torch.randn(8, 32, 4, 4), torch.randn(8, 32, 4, 4), torch.randn(8, 32, 4, 4)
        )


if __name__ == "__main__":
    unittest.main()
