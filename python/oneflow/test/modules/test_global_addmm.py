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


@autotest(n=1, check_graph=True)
def _test_addmm_with_random_data(test_case, placement, sbp):
    m = random(1, 3) * 8
    n = random(1, 3) * 8
    k = random(1, 3) * 8
    input = random_tensor(ndim=2, dim0=m, dim1=n).to_global(
        placement=placement, sbp=sbp
    )
    mat1 = random_tensor(ndim=2, dim0=m, dim1=k).to_global(placement=placement, sbp=sbp)
    mat2 = random_tensor(ndim=2, dim0=k, dim1=n).to_global(placement=placement, sbp=sbp)
    y = torch.addmm(
        input, mat1, mat2, beta=random().to(float), alpha=random().to(float),
    )
    return y


@autotest(n=1, check_graph=True)
def _test_addmm_broadcast_with_random_data(test_case, placement, sbp):
    m = random(1, 3) * 8
    n = random(1, 3) * 8
    k = random(1, 3) * 8
    input = random_tensor(ndim=2, dim0=1, dim1=1).to_global(
        placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))]
    )
    mat1 = random_tensor(ndim=2, dim0=m, dim1=k).to_global(placement=placement, sbp=sbp)
    mat2 = random_tensor(ndim=2, dim0=k, dim1=n).to_global(placement=placement, sbp=sbp)
    y = torch.addmm(
        input, mat1, mat2, beta=random().to(float), alpha=random().to(float),
    )
    return y


class TestAddmm(flow.unittest.TestCase):
    @globaltest
    def test_addmm(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_addmm_with_random_data(test_case, placement, sbp)
                _test_addmm_broadcast_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
