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
from random import randint
from random import choice

import numpy as np
from automated_test_util import *

import oneflow as flow
import oneflow.unittest


class TestAffineGrid(flow.unittest.TestCase):
    def test_affine_grid_2d(test_case):
        input = flow.Tensor(np.arange(1., 7).reshape((1, 2, 3)), dtype=flow.float32)
        output = flow.nn.functional.affine_grid(input, flow.Size([1, 1, 2, 2]), align_corners=True)
        groundtruth = np.array([[[[0., -3.], [2., 5.]], [[4., 7.], [6., 15.]]]])
        test_case.assertTrue(np.allclose(output.numpy(), groundtruth, rtol=1e-4, atol=1e-8))

        output = flow.nn.functional.affine_grid(input, flow.Size([1, 1, 2, 2]), align_corners=False)
        groundtruth = np.array([[[[1.5, 1.5], [2.5, 5.5]], [[3.5, 6.5], [4.5, 10.5]]]])
        test_case.assertTrue(np.allclose(output.numpy(), groundtruth, rtol=1e-4, atol=1e-8))

    def test_affine_grid_3d(test_case):
        input = flow.Tensor(np.arange(1., 13).reshape((1, 3, 4)), dtype=flow.float32)
        output = flow.nn.functional.affine_grid(input, flow.Size([1, 1, 2, 2, 2]), align_corners=True)
        groundtruth = np.array([[[[[-2., -10., -18.], [0., 0., 0.]], [[2., 2., 2.], [4., 12., 20.]]],
              [[[4., 4., 4.], [6., 14., 22.]], [[8., 16., 24.], [10., 26., 42.]]]]])
        test_case.assertTrue(np.allclose(output.numpy(), groundtruth, rtol=1e-4, atol=1e-8))

        output = flow.nn.functional.affine_grid(input, flow.Size([1, 1, 2, 2, 2]), align_corners=False)
        groundtruth = np.array([[[[[1., -1., -3.], [2., 4., 6.]], [[3., 5., 7.], [4., 10., 16.]]],
              [[[4., 6., 8.], [5., 11., 17.]], [[6., 12., 18.], [7., 17., 27.]]]]])
        test_case.assertTrue(np.allclose(output.numpy(), groundtruth, rtol=1e-4, atol=1e-8))

    @autotest()
    def test_flow_affine_grid_2d_with_random_data(test_case):
        N = randint(1, 8)
        C = randint(1, 8)
        H = randint(1, 8)
        W = randint(1, 8)
        device = random_device()
        align_corners = choice([True, False])
        theta = random_pytorch_tensor(ndim=3, dim0=N, dim1=2, dim2=3).to(device)
        output = torch.nn.functional.affine_grid(theta, (N, C, H, W), align_corners=align_corners).to(device)
        return output

    @autotest(rtol=1e-03, atol=1e-03)
    def test_flow_affine_grid_3d_with_random_data(test_case):
        N = randint(1, 8)
        C = randint(1, 8)
        D = randint(1, 8)
        H = randint(1, 8)
        W = randint(1, 8)
        device = random_device()
        align_corners = choice([True, False])
        theta = random_pytorch_tensor(ndim=3, dim0=N, dim1=3, dim2=4).to(device)
        output = torch.nn.functional.affine_grid(theta, (N, C, D, H, W), align_corners=align_corners).to(device)
        return output


if __name__ == "__main__":
    unittest.main()
