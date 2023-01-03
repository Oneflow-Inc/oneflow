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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True, rtol=0.5, atol=0.5)
def _test_sqrt_sum_with_cpu_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=4, dim0=8, dim1=32, dim2=40, dim3=64).to_global(
        placement=placement, sbp=sbp
    )
    y = torch.linalg.norm(x)
    return y


@autotest(n=1, check_graph=True, rtol=0.5, atol=0.5)
def _test_scalar_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=4, dim0=8, dim1=24, dim2=16, dim3=40).to_global(
        placement=placement, sbp=sbp
    )
    y = torch.linalg.norm(x)
    return y


class TestGlobalLinalgVectorNorm2D(flow.unittest.TestCase):
    @globaltest
    def test_sqrt_sum_with_cpu_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_sqrt_sum_with_cpu_random_data(test_case, placement, sbp)

    @globaltest
    def test_scalar_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_scalar_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
