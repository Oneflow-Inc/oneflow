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
from oneflow.test_utils.automated_test_util import *
import oneflow.unittest


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_less_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x1 = random_tensor(ndim, *dims)
    x1 = x1.to_global(placement=placement, sbp=sbp)
    x2 = random_tensor(ndim, *dims)
    x2 = x2.to_global(placement=placement, sbp=sbp)

    z = torch.lt(x1, x2)
    return z


class TestLessConsistent(flow.unittest.TestCase):
    @globaltest
    def test_less(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                _test_less_impl(test_case, ndim, placement, sbp)


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_less_equal_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x1 = random_tensor(ndim, *dims)
    x1 = x1.to_global(placement=placement, sbp=sbp)
    x2 = random_tensor(ndim, *dims)
    x2 = x2.to_global(placement=placement, sbp=sbp)

    z = torch.le(x1, x2)
    return z


class TestLessEqualConsistent(flow.unittest.TestCase):
    @globaltest
    def test_less_equal(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                _test_less_equal_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
