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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@autotest(n=1, rtol=1e-03, atol=1e-04, check_graph=True)
def _test_flow_grid_sample_cudnn(test_case, placement, sbp):
    # cudnn only support 4D input, with mode = 'bilinear' && padding_mode = 'zeros' && align_corners
    N = random(1, 3).to(int) * 8
    C = random(1, 3).to(int) * 8
    in_H = random(1, 8).to(int)
    in_W = random(1, 8).to(int)
    out_H = random(1, 8).to(int)
    out_W = random(1, 8).to(int)
    mode = "bilinear"
    padding_mode = "zeros"
    align_corners = True
    theta = random_tensor(ndim=3, dim0=N, dim1=2, dim2=3).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1)
    )
    grid = torch.nn.functional.affine_grid(
        theta, (N, C, out_H, out_W), align_corners=align_corners
    )
    input = random_tensor(ndim=4, dim0=N, dim1=C, dim2=in_H, dim3=in_W).to_global(
        placement=placement, sbp=sbp
    )
    output = torch.nn.functional.grid_sample(
        input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners,
    )
    return output


# This test may fail due to using ::floor in backward
# floor(1.99999988) = 1 and floor(2.000000) = 2, then select differente images pixel
@autotest(
    n=1,
    auto_backward=False,
    rtol=1e-03,
    atol=1e-04,
    check_graph=True,
    check_allclose=False,
)
def _test_flow_grid_sample_4d(test_case, placement, sbp):
    N = random(1, 3).to(int) * 8
    C = random(1, 3).to(int) * 8
    in_H = random(1, 8).to(int)
    in_W = random(1, 8).to(int)
    out_H = random(1, 8).to(int)
    out_W = random(1, 8).to(int)
    mode = oneof("bilinear", "nearest", "bicubic")
    padding_mode = oneof("zeros", "border", "reflection")
    align_corners = oneof(True, False)
    theta = random_tensor(ndim=3, dim0=N, dim1=2, dim2=3).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1)
    )
    grid = torch.nn.functional.affine_grid(
        theta, (N, C, out_H, out_W), align_corners=align_corners
    )
    input = random_tensor(ndim=4, dim0=N, dim1=C, dim2=in_H, dim3=in_W).to_global(
        placement=placement, sbp=sbp
    )
    output = torch.nn.functional.grid_sample(
        input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners,
    )
    return output


@autotest(n=1, auto_backward=False, rtol=1e-03, atol=1e-03, check_graph=True)
def _test_flow_grid_sample_5d(test_case, placement, sbp):
    N = random(1, 3).to(int) * 8
    C = random(1, 3).to(int) * 8
    in_D = random(1, 8).to(int)
    in_H = random(1, 8).to(int)
    in_W = random(1, 8).to(int)
    out_D = random(1, 8).to(int)
    out_H = random(1, 8).to(int)
    out_W = random(1, 8).to(int)
    mode = oneof("bilinear", "nearest")
    padding_mode = oneof("zeros", "border", "reflection")
    align_corners = oneof(True, False)
    theta = random_tensor(ndim=3, dim0=N, dim1=3, dim2=4).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1)
    )
    grid = torch.nn.functional.affine_grid(
        theta, (N, C, out_D, out_H, out_W), align_corners=align_corners
    )
    input = random_tensor(
        ndim=5, dim0=N, dim1=C, dim2=in_D, dim3=in_H, dim4=in_W
    ).to_global(placement=placement, sbp=sbp)
    output = torch.nn.functional.grid_sample(
        input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners,
    )
    return output


class TestGridSample(flow.unittest.TestCase):
    @globaltest
    def test_grid_sample(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                if placement.type == "cuda":
                    _test_flow_grid_sample_cudnn(test_case, placement, sbp)
                _test_flow_grid_sample_4d(test_case, placement, sbp)
                _test_flow_grid_sample_5d(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
