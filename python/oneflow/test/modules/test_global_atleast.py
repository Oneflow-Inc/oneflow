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
def _test_atleast1d_with_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=1, dim0=8).to_global(placement, sbp)
    y = random_tensor(ndim=2, dim0=8).to_global(placement, sbp)
    out = torch.atleast_1d([x, y])
    return out


@autotest(n=1, check_graph=True)
def _test_atleast2d_with_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=1, dim0=8).to_global(placement, sbp)
    y = random_tensor(ndim=2, dim0=8).to_global(placement, sbp)
    z = random_tensor(ndim=3, dim0=8).to_global(placement, sbp)
    out = torch.atleast_2d([x, y, z])
    return out


@autotest(n=1, check_graph=True)
def _test_atleast3d_with_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=1, dim0=8).to_global(placement, sbp)
    y = random_tensor(ndim=2, dim0=8).to_global(placement, sbp)
    z = random_tensor(ndim=3, dim0=8).to_global(placement, sbp)
    p = random_tensor(ndim=4, dim0=8).to_global(placement, sbp)
    out = torch.atleast_3d([x, y, z, p])
    return out


class TestAtLeastModule(flow.unittest.TestCase):
    @globaltest
    def test_atleast1d_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_atleast1d_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_atleast2d_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_atleast2d_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_atleast3d_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_atleast3d_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
