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


class TestSplit(flow.unittest.TestCase):
    @autotest()
    def test_flow_split_with_random_data(test_case):
        k0 = random(2, 6)
        k1 = random(2, 6)
        k2 = random(2, 6)
        rand_dim = random(0, 3).to(int)
        device = random_device()
        x = random_pytorch_tensor(ndim=3, dim0=k0, dim1=k1, dim3=k2).to(device)
        res = torch.split(x, split_size_or_sections=2, dim=rand_dim)
        return torch.cat(res, rand_dim)

    @autotest()
    def test_flow_split_sizes_with_random_data(test_case):
        k0 = random(2, 6)
        k1 = 7
        k2 = random(2, 6)
        device = random_device()
        x = random_pytorch_tensor(ndim=3, dim0=k0, dim1=k1, dim3=k2).to(device)
        res = torch.split(x, split_size_or_sections=[1, 2, 3, 1], dim=1)
        return torch.cat(res, dim=1)


if __name__ == "__main__":
    unittest.main()
