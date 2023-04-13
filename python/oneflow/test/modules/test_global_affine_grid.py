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


@autotest(n=1, rtol=1e-03, atol=1e-04, check_graph=True)
def _test_affine_grid_2d_with_random_data(test_case, placement, sbp):
    N = random(1, 3).to(int).value() * 8
    C = random(1, 8).to(int).value()
    H = random(1, 8).to(int).value()
    W = random(1, 8).to(int).value()
    align_corners = oneof(True, False).value()
    dims = [N, 2, 3]

    theta = random_tensor(3, *dims).to_global(placement=placement, sbp=sbp)
    output = torch.nn.functional.affine_grid(
        theta, (N, C, H, W), align_corners=align_corners
    )
    return output


@autotest(n=1, rtol=1e-03, atol=1e-04, check_graph=True)
def _test_affine_grid_3d_with_random_data(test_case, placement, sbp):
    N = random(1, 3).to(int) * 8
    C = random(1, 8).to(int)
    D = random(1, 8).to(int)
    H = random(1, 8).to(int)
    W = random(1, 8).to(int)
    align_corners = oneof(True, False)
    dims = [N, 3, 4]

    theta = random_tensor(3, *dims).to_global(placement=placement, sbp=sbp)
    output = torch.nn.functional.affine_grid(
        theta, (N, C, D, H, W), align_corners=align_corners
    )
    return output


class TestAffineGrid(flow.unittest.TestCase):
    @globaltest
    def test_affine_grid(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_affine_grid_2d_with_random_data(test_case, placement, sbp)
                _test_affine_grid_3d_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
