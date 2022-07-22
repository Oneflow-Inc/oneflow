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


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_sub(test_case, placement, sbp):
    x = random_tensor(2, 8, 8).to_global(placement=placement, sbp=sbp)
    y = random_tensor(2, 8, 8).to_global(placement=placement, sbp=sbp)
    out1 = x - y
    out2 = x - 2
    out3 = 2 - x
    out4 = torch.sub(x, y)
    return out1, out2, out3, out4


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_sub_with_0_size_data(test_case, placement, sbp):
    device = random_device()
    x = random_tensor(2, 0, 8).to_global(placement=placement, sbp=sbp)
    out1 = x - 2
    out2 = 2 - x
    return out1, out2


class TestGlobalSubModule(flow.unittest.TestCase):
    @globaltest
    def test_global_sub(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_sub(test_case, placement, sbp)

    @globaltest
    def test_global_sub_with_0_size_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2, valid_split_axis=1):
                _test_global_sub_with_0_size_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
