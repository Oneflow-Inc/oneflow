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
def _test_logical_binary_impl(test_case, ndim, placement, sbp, f):
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x1 = random_tensor(ndim, *dims)
    x2 = x1.to_global(placement=placement, sbp=sbp)
    y1 = random_tensor(ndim, *dims)
    y2 = y1.to_global(placement=placement, sbp=sbp)
    z = f(x2, y2)
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_logical_not_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x1 = random_tensor(ndim, *dims)
    x2 = x1.to_global(placement=placement, sbp=sbp)
    z = torch.logical_not(x2)
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_logical_reduce_impl(test_case, ndim, placement, sbp, f):
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x1 = random_tensor(ndim, *dims, requires_grad=False)
    x2 = x1.to_global(placement=placement, sbp=sbp)
    z = f(x2)
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_logical_reduce_with_dim_impl(test_case, ndim, placement, sbp, f):
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x1 = random_tensor(ndim, *dims, requires_grad=False)
    x2 = x1.to_global(placement=placement, sbp=sbp)
    z = f(x2, random(1, ndim).to(int), keepdim=oneof(True, False))
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_logical_reduce_keepdim_impl(test_case, placement, sbp, f):
    dims = [random(1, 4) * 8 for i in range(4)]
    x1 = random_tensor(4, *dims, requires_grad=False)
    x2 = x1.to_global(placement=placement, sbp=sbp)
    z = f(x2, 1, keepdim=oneof(True, False))
    return z


class TestLogicalConsistent(flow.unittest.TestCase):
    @globaltest
    def test_logical_binary(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                for f in [torch.logical_and, torch.logical_or, torch.logical_xor]:
                    _test_logical_binary_impl(test_case, ndim, placement, sbp, f)

    @globaltest
    def test_logical_not(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_logical_not_impl(test_case, ndim, placement, sbp)

    @globaltest
    def test_logical_reduce(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                for f in [torch.all, torch.any]:
                    _test_logical_reduce_impl(test_case, ndim, placement, sbp, f)
                    _test_logical_reduce_with_dim_impl(
                        test_case, ndim, placement, sbp, f
                    )
                    _test_logical_reduce_keepdim_impl(test_case, placement, sbp, f)


if __name__ == "__main__":
    unittest.main()
