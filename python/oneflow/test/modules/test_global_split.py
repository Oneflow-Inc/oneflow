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


@autotest(n=2, check_graph=True)
def _test_flow_split_with_random_data(test_case, placement, sbp):
    k0 = random(2, 6) * 8
    k1 = random(2, 6) * 8
    k2 = random(2, 6) * 8
    rand_dim = random(0, 3).to(int)
    x = random_tensor(ndim=3, dim0=k0, dim1=k1, dim2=k2).to_global(
        placement=placement, sbp=sbp
    )
    res = torch.split(x, 2, dim=rand_dim)
    return torch.cat(res, rand_dim)


@autotest(n=2, check_graph=True)
def _test_flow_split_sizes_with_random_data(test_case, placement, sbp):
    k0 = random(2, 6) * 8
    k1 = 16
    k2 = random(2, 6) * 8
    x = random_tensor(ndim=3, dim0=k0, dim1=k1, dim2=k2).to_global(
        placement=placement, sbp=sbp
    )
    res = torch.split(x, [6, 3, 4, 3], dim=1)
    return torch.cat(res, dim=1)


@autotest(n=2, check_graph=True)
def _test_flow_split_sizes_neg_dim_with_random_data(test_case, placement, sbp):
    k0 = random(2, 6) * 8
    k1 = 16
    k2 = random(2, 6) * 8
    x = random_tensor(ndim=3, dim0=k0, dim1=k1, dim2=k2).to_global(
        placement=placement, sbp=sbp
    )
    res = torch.split(x, [6, 3, 4, 3], dim=-2)
    return torch.cat(res, dim=1)


class TestGlobalSplitModule(flow.unittest.TestCase):
    @globaltest
    def test_flow_split_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_split_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_flow_split_sizes_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_split_sizes_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_flow_split_sizes_neg_dim_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_split_sizes_neg_dim_with_random_data(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()
