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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=False)
def _test_global_nansum_against_pytorch(test_case, placement, sbp):
    x = random_tensor(4, 8, 16, 8, 24).to_global(placement, sbp)
    mask = x < 0
    x = x.masked_fill(mask, float("nan"))
    y = torch.nansum(x)
    return y


@autotest(n=1, check_graph=False)
def _test_global_nansum_with_0_size_tensor(test_case, placement, sbp):
    x = random_tensor(4, 8, 16, 0, 24).to_global(placement, sbp)
    mask = torch.ones_like(x).bool()
    x = x.masked_fill(mask, float("nan"))
    y = torch.nansum(x, dim=random(0, 3).to(int))
    return y


class TestGlobalNanSumModule(flow.unittest.TestCase):
    @globaltest
    def test_global_nansum_against_pytorch(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_global_nansum_against_pytorch(test_case, placement, sbp)

    @globaltest
    def test_global_nansum_with_0_size_tensor(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4, valid_split_axis=[0, 1, 3]):
                _test_global_nansum_with_0_size_tensor(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
