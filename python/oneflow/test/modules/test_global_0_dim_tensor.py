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
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True)
def _test_0_dim_tensor(test_case, placement, sbp):
    x1 = random_tensor(0).to_global(placement=placement, sbp=sbp)
    x2 = random_tensor(0).to_global(placement=placement, sbp=sbp)
    y1 = x1 * x2
    y2 = x1 + x2
    return y1 + y2


@autotest(n=1, check_graph=True)
def _test_1dim_slice(test_case, placement, sbp):
    x = random_tensor(1, random(1, 4) * 8).to_global(placement=placement, sbp=sbp)
    return x[5]


class TestZeroDimensionTensor(flow.unittest.TestCase):
    @globaltest
    def test_0_dim_tensor(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=0):
                _test_0_dim_tensor(test_case, placement, sbp)
            for sbp in all_sbp(placement, max_dim=1):
                _test_1dim_slice(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
