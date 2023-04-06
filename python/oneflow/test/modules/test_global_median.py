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
import torch
from functools import reduce
import operator
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True)
def _test_median(test_case, placement, sbp, ndim):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    return torch.median(x)


@autotest(n=1, check_graph=True)
def _test_median_with_indices(test_case, placement, sbp, ndim):
    dim = random(1, ndim).to(int).value()
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = choice_tensor(
        reduce(operator.mul, dim_list, 1),
        dim_list,
        replace=False,
        dtype=float,
        requires_grad=True,
    ).to_global(placement, sbp)
    return torch.median(x, dim)


class TestMedian(flow.unittest.TestCase):
    @globaltest
    def test_median(test_case):
        ndim = random(2, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_median(test_case, placement, sbp, ndim)
                _test_median_with_indices(test_case, placement, sbp, ndim)


if __name__ == "__main__":
    unittest.main()
