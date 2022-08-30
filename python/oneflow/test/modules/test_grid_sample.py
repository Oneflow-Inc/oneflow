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
class TestGridSample(flow.unittest.TestCase):
    def test_grid_sample_4d(test_case):
        input = flow.tensor(
            np.arange(1.0, 11).reshape((1, 1, 2, 5)), dtype=flow.float32
        )
        np_grid = np.array(
            [
                [[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
                [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]],
            ]
        ).reshape(1, 2, 5, 2)
        grid = flow.tensor(np_grid, dtype=flow.float32)
        groundtruth = np.reshape(
            np.array([[0.0, 8.0, 5.0, 7.0, 9.0], [1.0, 8.0, 5.0, 8.0, 0.0]]),
            (1, 1, 2, 5),
        )
        output = flow.nn.functional.grid_sample(
            input, grid, mode="nearest", padding_mode="zeros", align_corners=True
        )
        test_case.assertTrue(
            np.allclose(output.numpy(), groundtruth, rtol=1e-3, atol=1e-4)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(rtol=1e-03, atol=1e-04, check_graph=True)
    def test_flow_grid_sample_cudnn_with_random_data(test_case):
        # cudnn only support 4D input, with mode = 'bilinear' && padding_mode = 'zeros' && align_corners
        N = randint(1, 8)
        C = randint(1, 8)
        in_H = randint(1, 8)
        in_W = randint(1, 8)
        out_H = randint(1, 8)
        out_W = randint(1, 8)
        device = "cuda"
        mode = "bilinear"
        padding_mode = "zeros"
        align_corners = True
        theta = random_tensor(ndim=3, dim0=N, dim1=2, dim2=3).to(device)
        grid = torch.nn.functional.affine_grid(
            theta, (N, C, out_H, out_W), align_corners=align_corners
        ).to(device)
        input = random_tensor(ndim=4, dim0=N, dim1=C, dim2=in_H, dim3=in_W).to(device)
        output = torch.nn.functional.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return output

    # This test may fail due to using ::floor in backward
    # floor(1.99999988) = 1 and floor(2.000000) = 2, then select differente images pixel
    @autotest(
        auto_backward=False,
        rtol=1e-03,
        atol=1e-04,
        check_graph=True,
        check_allclose=False,
    )
    def test_flow_grid_sample_4d_with_random_data(test_case):
        N = randint(1, 8)
        C = randint(1, 8)
        in_H = randint(1, 8)
        in_W = randint(1, 8)
        out_H = randint(1, 8)
        out_W = randint(1, 8)
        device = random_device()
        mode = choice(["bilinear", "nearest", "bicubic"])
        padding_mode = choice(["zeros", "border", "reflection"])
        align_corners = choice([True, False])
        theta = random_tensor(ndim=3, dim0=N, dim1=2, dim2=3).to(device)
        grid = torch.nn.functional.affine_grid(
            theta, (N, C, out_H, out_W), align_corners=align_corners
        ).to(device)
        input = random_tensor(ndim=4, dim0=N, dim1=C, dim2=in_H, dim3=in_W).to(device)
        output = torch.nn.functional.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return output

    @autotest(auto_backward=False, rtol=1e-03, atol=1e-03, check_graph=True)
    def test_flow_grid_sample_5d_with_random_data(test_case):
        N = randint(1, 8)
        C = randint(1, 8)
        in_D = randint(1, 8)
        in_H = randint(1, 8)
        in_W = randint(1, 8)
        out_D = randint(1, 8)
        out_H = randint(1, 8)
        out_W = randint(1, 8)
        device = random_device()
        mode = choice(["bilinear", "nearest"])
        padding_mode = choice(["zeros", "border", "reflection"])
        align_corners = choice([True, False])
        theta = random_tensor(ndim=3, dim0=N, dim1=3, dim2=4).to(device)
        grid = torch.nn.functional.affine_grid(
            theta, (N, C, out_D, out_H, out_W), align_corners=align_corners
        ).to(device)
        input = random_tensor(
            ndim=5, dim0=N, dim1=C, dim2=in_D, dim3=in_H, dim4=in_W
        ).to(device)
        output = torch.nn.functional.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return output

    @profile(torch.nn.functional.grid_sample)
    def profile_grid_sample(test_case):
        input = torch.ones(32, 3, 128, 128)
        grid = torch.ones(32, 64, 64, 2)
        torch.nn.functional.grid_sample(input, grid)
        torch.nn.functional.grid_sample(input, grid, align_corners=True)
        torch.nn.functional.grid_sample(input, grid, mode="nearest", align_corners=True)
        torch.nn.functional.grid_sample(input, grid, mode="bicubic", align_corners=True)
        torch.nn.functional.grid_sample(input, grid, padding_mode="border")
        torch.nn.functional.grid_sample(input, grid, padding_mode="reflection")


if __name__ == "__main__":
    unittest.main()
