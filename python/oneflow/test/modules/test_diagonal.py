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
import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestDiagonal(flow.unittest.TestCase):
    @autotest(n=10, check_graph=True)
    def test_flow_diagonal_with_random_data(test_case):
        device = random_device()
        offset = random(-5, 5).to(int)
        dim1 = random(-4, 4).to(int)
        dim2 = random(-4, 4).to(int)

        x = random_tensor(
            ndim=4,
            dim1=random(4, 6),
            dim2=random(4, 6),
            dim3=random(4, 6),
            dim4=random(4, 6),
        ).to(device)
        z = torch.diagonal(x, offset, dim1, dim2)
        return z

    @autotest(auto_backward=False, n=10, check_graph=True)
    def test_flow_diagonal_with_random_data(test_case):
        device = random_device()
        offset = random(-5, 5).to(int)
        dim1 = random(-4, 4).to(int)
        dim2 = random(-4, 4).to(int)

        x = random_tensor(
            ndim=4,
            dim1=random(4, 6),
            dim2=random(4, 6),
            dim3=random(4, 6),
            dim4=random(4, 6),
        ).to(device, torch.bool)
        z = torch.diagonal(x, offset, dim1, dim2)
        return z

    @profile(torch.diagonal)
    def profile_diagonal(test_case):
        input1 = torch.ones(128, 128)
        input2 = torch.ones(16, 10, 128, 128)
        torch.diagonal(input1, 0)
        torch.diagonal(input1, 1)
        torch.diagonal(input2, offset=-1, dim1=1, dim2=2)


if __name__ == "__main__":
    unittest.main()
