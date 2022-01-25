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
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *

@autotest(n=10, check_graph=False)
def test_flow_diagonal_impl(test_case, placement, sbp):
    device = random_device()
    offset = random(-5, 5).to(int)
    dim1 = random(-4, 4).to(int)
    dim2 = random(-4, 4).to(int)

    x = random_pytorch_tensor(
        ndim=4,
        dim1=random(1, 6)*8,
        dim2=random(1, 6)*8,
        dim3=random(1, 6)*8,
        dim4=random(1, 6)*8,
    )
    y = x.to_consistent(placement=placement, sbp=sbp)
    z = torch.diagonal(y, offset, dim1, dim2)
    return z

class TestDiagonalConsistent(flow.unittest.TestCase):
    @consistent
    def test_flow_diagonal_impl(test_case):
        # random ndim in range [1,5]
        ndim = np.random.randint(1, 6)
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                test_flow_diagonal_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
