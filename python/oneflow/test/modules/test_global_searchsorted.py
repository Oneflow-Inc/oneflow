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


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_search_sorted(test_case, placement, sbp, ndim):
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    sorted_sequence = random_tensor(ndim, *dims).to_global(placement, sbp)
    values = random_tensor(ndim, *dims).to_global(placement, sbp)
    y = torch.searchsorted(
        sorted_sequence, values, out_int32=oneof(True, False), right=oneof(True, False),
    )
    return y


class TestSearchSorted_Global(flow.unittest.TestCase):
    @globaltest
    def test_search_sorted(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_search_sorted(test_case, placement, sbp, ndim)


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_search_sorted_scalar(test_case, placement, sbp):
    dim0 = [random(1, 3) * 8]
    sorted_sequence = random_tensor(1, *dim0).to_global(placement, sbp)
    values = 5
    y = torch.searchsorted(
        sorted_sequence, values, out_int32=oneof(True, False), right=oneof(True, False),
    )
    return y


class TestSearchSortedScalar_Global(flow.unittest.TestCase):
    @globaltest
    def test_search_sorted_scalar(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_search_sorted_scalar(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
