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

class TestDiagonal(flow.unittest.TestCase):
    @autotest()
    def test_flow_diagonal_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(
            ndim=4,
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
            dim4=random(3, 6),
        ).to(device)
        z = torch.diagonal(x, 1, 0, 2)
        return z

if __name__ == "__main__":
    unittest.main()