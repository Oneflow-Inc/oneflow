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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True)
def _test_cat_with_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_global(placement=placement, sbp=sbp)
    return torch.cat((x, x), random(0, 2).to(int))


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_concat_with_input_0_size_data(test_case, placement, sbp):
    x = random_tensor(4, 8, 8, 2, 4).to_global(placement=placement, sbp=sbp)
    y = random_tensor(4, 8, 8, random(0, 3) * 8, 4).to_global(
        placement=placement, sbp=sbp
    )
    z = torch.cat((x, y), dim=2)
    return z


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_concat_with_output_0_size_data(test_case, placement, sbp):
    x = random_tensor(4, 8, 8, 0, 4).to_global(placement=placement, sbp=sbp)
    y = random_tensor(4, 8, 8, 0, 4).to_global(placement=placement, sbp=sbp)
    z = torch.cat((x, y), dim=2)
    return z


@autotest(n=1, check_graph=True)
def _test_cat_only_one_tensor(test_case, placement, sbp):
    x = random_tensor(4, 8, 8, random(1, 3) * 8, 8).to_global(
        placement=placement, sbp=sbp
    )
    return torch.cat((x,), 0)


class TestModule(flow.unittest.TestCase):
    @globaltest
    def test_cat_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_cat_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_cat_only_one_tensor(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_cat_only_one_tensor(test_case, placement, sbp)

    @globaltest
    def test_concat_with_input_0_size_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_concat_with_input_0_size_data(test_case, placement, sbp)

    @globaltest
    def test_concat_with_output_0_size_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_concat_with_output_0_size_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
