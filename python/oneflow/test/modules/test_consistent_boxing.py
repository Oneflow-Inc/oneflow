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


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_boxing_with_random_data(test_case, ndim, placement, sbp_in, sbp_out):
    # NOTE: Consuming a lot of time for transferring. Should use a tensor as small as possible.
    dims = [8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    # NOTE: Boxing collector (a.k.a. middle nodes algorithm) do not support transferring a 1D sbp to nd sbp at this moment.
    # We do not support B -> (S(0), S(1)) for lazy.
    # Thus, we transfer B to (B, B).
    # TODO: Support 1d to nd sbp transfer using middle nodes.
    x = x.to_global(
        placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
    )

    # print("x sbp: ", x.sbp)
    # print("sbp in: ", sbp_in)
    y = x.to_global(placement=placement, sbp=sbp_in)
    # print("sbp out: ", sbp_out)
    z = y.to_global(sbp=sbp_out)
    return z


class TestConsistentSplitModule(flow.unittest.TestCase):
    @global_view
    def test_boxing_with_random_data(test_case):
        for ndim in range(2, 3):
            for placement in all_placement():
                if len(placement.hierarchy) != 2 or min(placement.hierarchy) <= 1:
                    continue
                for sbp_in in all_sbp(placement, max_dim=ndim):
                    for sbp_out in all_sbp(placement, max_dim=ndim):
                        _test_boxing_with_random_data(
                            test_case, ndim, placement, sbp_in, sbp_out
                        )


if __name__ == "__main__":
    unittest.main()
