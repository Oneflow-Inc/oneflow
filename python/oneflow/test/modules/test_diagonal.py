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
        ndims = 4
        dim1 = 0
        dim2 = 0
        p_dim1 = 0
        p_dim2 = 0
        while p_dim1 == p_dim2:
            dim1 = random(-4, 4).to(int).value()
            dim2 = random(-4, 4).to(int).value()
            p_dim1 = dim1 if dim1 >= 0 else dim1 + ndims;
            p_dim2 = dim2 if dim2 >= 0 else dim2 + ndims;

        x = random_tensor(
            ndim=ndims,
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
        ndims = 4
        dim1 = 0
        dim2 = 0
        p_dim1 = 0
        p_dim2 = 0
        while p_dim1 == p_dim2:
            dim1 = random(-4, 4).to(int).value()
            dim2 = random(-4, 4).to(int).value()
            p_dim1 = dim1 if dim1 >= 0 else dim1 + ndims;
            p_dim2 = dim2 if dim2 >= 0 else dim2 + ndims;

        x = random_tensor(
            ndim=ndims,
            dim1=random(4, 6),
            dim2=random(4, 6),
            dim3=random(4, 6),
            dim4=random(4, 6),
        ).to(device, torch.bool)
        z = torch.diagonal(x, offset, dim1, dim2)
        return z


if __name__ == "__main__":
    unittest.main()
