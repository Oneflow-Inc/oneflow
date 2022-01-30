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


@autotest(n=3, auto_backward=False, check_graph=True)
def test_greater_impl(test_case, ndim, placement):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    # NOTE: Boxing collector (a.k.a. middle nodes algorithm) do not support transferring a 1D sbp to nd sbp at this moment.
    # We do not support B -> (S(0), S(1)) for lazy.
    # Thus, we transfer B to (B, B).
    # TODO: Support 1d to nd sbp transfer using middle nodes.
    x = x.to_consistent(placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast])

    x1 = x.to_consistent(placement=placement, sbp=[flow.sbp.split(0), flow.sbp.split(1)])
    # print("x1 sbp: ", x1.sbp)
    x2 = x.to_consistent(placement=placement, sbp=[flow.sbp.split(1), flow.sbp.split(0)])
    # print("x2 sbp: ", x2.sbp)

    y = x1 + x2
    # print("y sbp: ", y.sbp)
    return y


class TestGreaterConsistent(flow.unittest.TestCase):
    @consistent
    def test_greater(test_case):
        # random ndim in range [1,4]
        ndim = random(2, 3).to(int).value()
        for placement in all_placement():
            if len(placement.hierarchy) != 2 or min(placement.hierarchy) <= 1:
                continue
            test_greater_impl(test_case, ndim, placement)


if __name__ == "__main__":
    unittest.main()
