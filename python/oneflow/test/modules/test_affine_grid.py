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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestAffineGrid(flow.unittest.TestCase):
    def test_affine_grid_2d(test_case):
        input = flow.tensor(np.arange(1.0, 7).reshape((1, 2, 3)), dtype=flow.float32)
        output = flow.nn.functional.affine_grid(
            input, flow.Size([1, 1, 2, 2]), align_corners=True
        )
        groundtruth = np.array([[[[0.0, -3.0], [2.0, 5.0]], [[4.0, 7.0], [6.0, 15.0]]]])
        test_case.assertTrue(
            np.allclose(output.numpy(), groundtruth, rtol=1e-3, atol=1e-4)
        )

        output = flow.nn.functional.affine_grid(
            input, flow.Size([1, 1, 2, 2]), align_corners=False
        )
        groundtruth = np.array([[[[1.5, 1.5], [2.5, 5.5]], [[3.5, 6.5], [4.5, 10.5]]]])
        test_case.assertTrue(
            np.allclose(output.numpy(), groundtruth, rtol=1e-3, atol=1e-4)
        )

    def test_affine_grid_3d(test_case):
        input = flow.tensor(np.arange(1.0, 13).reshape((1, 3, 4)), dtype=flow.float32)
        output = flow.nn.functional.affine_grid(
            input, flow.Size([1, 1, 2, 2, 2]), align_corners=True
        )
        groundtruth = np.array(
            [
                [
                    [
                        [[-2.0, -10.0, -18.0], [0.0, 0.0, 0.0]],
                        [[2.0, 2.0, 2.0], [4.0, 12.0, 20.0]],
                    ],
                    [
                        [[4.0, 4.0, 4.0], [6.0, 14.0, 22.0]],
                        [[8.0, 16.0, 24.0], [10.0, 26.0, 42.0]],
                    ],
                ]
            ]
        )
        test_case.assertTrue(
            np.allclose(output.numpy(), groundtruth, rtol=1e-3, atol=1e-4)
        )

        output = flow.nn.functional.affine_grid(
            input, flow.Size([1, 1, 2, 2, 2]), align_corners=False
        )
        groundtruth = np.array(
            [
                [
                    [
                        [[1.0, -1.0, -3.0], [2.0, 4.0, 6.0]],
                        [[3.0, 5.0, 7.0], [4.0, 10.0, 16.0]],
                    ],
                    [
                        [[4.0, 6.0, 8.0], [5.0, 11.0, 17.0]],
                        [[6.0, 12.0, 18.0], [7.0, 17.0, 27.0]],
                    ],
                ]
            ]
        )
        test_case.assertTrue(
            np.allclose(output.numpy(), groundtruth, rtol=1e-3, atol=1e-4)
        )

    @autotest(n=5, rtol=1e-03, atol=1e-04, check_allclose=False, check_graph=True)
    def test_flow_affine_grid_2d_with_random_data(test_case):
        N = randint(1, 8)
        C = randint(1, 8)
        H = randint(1, 8)
        W = randint(1, 8)
        device = random_device()
        align_corners = choice([True, False])
        theta = random_tensor(ndim=3, dim0=N, dim1=2, dim2=3).to(device)
        output = torch.nn.functional.affine_grid(
            theta, (N, C, H, W), align_corners=align_corners
        ).to(device)
        return output

    @autotest(rtol=1e-03, atol=1e-03, check_allclose=False, check_graph=True)
    def test_flow_affine_grid_3d_with_random_data(test_case):
        N = randint(1, 8)
        C = randint(1, 8)
        D = randint(1, 8)
        H = randint(1, 8)
        W = randint(1, 8)
        device = random_device()
        align_corners = choice([True, False])
        theta = random_tensor(ndim=3, dim0=N, dim1=3, dim2=4).to(device)
        output = torch.nn.functional.affine_grid(
            theta, (N, C, D, H, W), align_corners=align_corners
        ).to(device)
        return output

    @profile(torch.nn.functional.affine_grid)
    def profile_affine_grid(test_case):
        input = torch.tensor(np.arange(1.0, 7).reshape((1, 2, 3)), dtype=torch.float32)
        torch.nn.functional.affine_grid(
            input, torch.Size([1, 1, 2, 2]), align_corners=True
        )


if __name__ == "__main__":
    unittest.main()
