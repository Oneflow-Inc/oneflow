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


@autotest(n=1, check_graph=False)
def _test_gather(test_case, placement, sbp):
    input = random_tensor(ndim=4, dim0=8, dim1=8, dim2=8, dim3=8).to_global(
        placement=placement, sbp=sbp
    )
    dim = random(0, 4).to(int)
    index = random_tensor(
        ndim=4,
        dim1=random(1, 3).to(int),
        dim2=random(1, 4).to(int),
        dim3=random(1, 5).to(int),
        dtype=int,
    ).to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))])
    return torch.gather(input, dim, index)


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_gather_bool(test_case, placement, sbp):
    input = (
        random_tensor(ndim=4, dim0=8, dim1=8, dim2=8, dim3=8)
        .to(torch.bool)
        .to_global(placement=placement, sbp=sbp)
    )
    dim = random(0, 4).to(int)
    index = random_tensor(
        ndim=4,
        dim1=random(1, 3).to(int),
        dim2=random(1, 4).to(int),
        dim3=random(1, 5).to(int),
        dtype=int,
    ).to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))])
    return torch.gather(input, dim, index)


class TestGather(flow.unittest.TestCase):
    @globaltest
    def test_gather(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_gather(test_case, placement, sbp)
                _test_gather_bool(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
