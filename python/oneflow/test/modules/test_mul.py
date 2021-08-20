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

import numpy as np
from automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestMulModule(flow.unittest.TestCase):
    @autotest()
    def test_mul_with_random_data_integer(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=1).to(device)
        k = random(10,100).to(int)
        y = torch.mul(x, k)
        return y
    
    @autotest()
    def test_mul_with_random_data_scalar(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4).to(device)
        y = random_pytorch_tensor(ndim=4).to(device)
        z = torch.mul(x, y)
        return z

    @autotest()
    def test_mul_with_random_data_broadcast(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=1).to(device)
        y = random_pytorch_tensor(ndim=1, dim0=4).to(device)
        z = torch.mul(x, y)
        return z

if __name__ == "__main__":
    unittest.main()
