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
from automated_test_util import *
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestSub(flow.unittest.TestCase):
    @autotest()
    def test_sub_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim0=0, dim1=3).to(device)
        y = random_pytorch_tensor(ndim=2, dim0=1, dim1=3).to(device)
        z = torch.sub(x, y)
        return z


    @autotest()
    def test_sub_with_scalar_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2).to(device)
        y = random_pytorch_tensor(ndim=2).to(device)
        z = torch.sub(x, y)
        return z

    @autotest()
    def test_sub_with_broadcast_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=5, dim0=1, dim1=4, dim2=1).to(device)
        y = random_pytorch_tensor(ndim=3, dim0=1, dim1=1).to(device)
        z = torch.sub(x, y)
        return z

if __name__ == "__main__":
    unittest.main()
