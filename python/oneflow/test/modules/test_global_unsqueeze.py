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


@autotest(n=1, check_graph=True)
def _test_flow_unsqueeze_with_random_data(test_case, placement, sbp):
    x = random_tensor(2, 8, 16).to_global(placement, sbp)
    y = torch.unsqueeze(x, random(0, 3).to(int))
    return y


@autotest(n=1, check_graph=True)
def _test_tensor_unsqueeze_with_random_data(test_case, placement, sbp):
    x = random_tensor(2, 8, 16).to_global(placement, sbp)
    y = x.unsqueeze(random(0, 3).to(int))
    return y


@autotest(n=1, check_graph=True)
def _test_unsqueeze_with_0_size_data(test_case, placement, sbp):
    x = random_tensor(3, 8, 16, 0).to_global(placement, sbp)
    y = torch.unsqueeze(x, random(0, 4).to(int))
    return y


class TestGlobalUnsqueeze(flow.unittest.TestCase):
    @globaltest
    def test_flow_unsqueeze_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_unsqueeze_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_tensor_unsqueeze_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_tensor_unsqueeze_with_random_data(test_case, placement, sbp)

    @globaltest
    def test_unsqueeze_with_0_size_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_unsqueeze_with_0_size_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
