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
class TestModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_matmul_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        z = torch.matmul(x, y)
        return z

    @autotest(check_graph=True)
    def test_flow_tensor_matmul_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        return x.matmul(y)

    @autotest(check_graph=True)
    def test_flow_tensor_broadcast_matmul_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=4, dim3=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        return x.matmul(y)

    @autotest(check_graph=True)
    def test_flow_mm_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        z = torch.mm(x, y)
        return z

    def test_flow_mv_with_random_data(test_case):
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=1, dim0=k).to(device)
        z = torch.mv(x, y)
        return z

    @profile(torch.mv)
    def profile_mv(test_case):
        torch.mv(torch.ones(32, 64), torch.ones(64))


if __name__ == "__main__":
    unittest.main()
