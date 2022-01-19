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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@autotest(auto_backward=False, check_graph=False)
def _test_squeeze_1d_input(test_case, placement, sbp):
    x = random_pytorch_tensor(1, 10, dtype=float).to_consistent(placement, sbp)
    y = torch.squeeze(x)
    return y


@autotest(auto_backward=False, check_graph=False)
def _test_squeeze_backward(test_case, placement, sbp):
    x = random_pytorch_tensor(
        4, 1, 1, 1, 16, dtype=float, requires_grad=True
    ).to_consistent(placement, sbp)
    y = torch.squeeze(x, dim=1).sum()
    y.backward()
    x_grad = x.grad
    return x


@autotest(auto_backward=False, check_graph=False)
def _test_flow_squeeze_with_random_data(test_case, placement, sbp):
    x = random_pytorch_tensor(2, 8, 16).to_consistent(placement, sbp)
    y = torch.squeeze(x, random(1, 3).to(int))
    return y


@autotest(auto_backward=False, check_graph=False)
def _test_squeeze_with_0_size_data(test_case, placement, sbp):
    x = random_pytorch_tensor(3, 8, 16, 0).to_consistent(placement, sbp)
    y = torch.squeeze(x)
    return y


class TestConsistentSqueeze(flow.unittest.TestCase):
    @consistent
    def test_squeeze_1d_input(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_squeeze_1d_input(test_case, placement, sbp)

    # File "/home/hanbinbin/oneflow/python/oneflow/test_utils/automated_test_util/torch_flow_dual_object.py", line 626, in __getattr__
    # assert (
    # AssertionError: pytorch value is None for attr grad, but oneflow is not.
    # @consistent
    # def test_squeeze_backward(test_case):
    #     for placement in all_placement():
    #         for sbp in all_sbp(placement, max_dim=4, valid_split_axis=3):
    #             if flow.env.get_rank() == 0:
    #                 print(placement, sbp)
    #             _test_squeeze_backward(test_case, placement, sbp)

    @consistent
    def test_flow_squeeze_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_squeeze_with_random_data(test_case, placement, sbp)

    # TODO(): Support 0 size tensor for eager consistent
    # @consistent
    # def test_squeeze_with_0_size_data(test_case):
    #     for placement in all_placement():
    #         for sbp in all_sbp(placement, max_dim=2):
    #             _test_squeeze_with_0_size_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
