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
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.nn.functional as F
import oneflow.unittest


@autotest(n=1, check_graph=False)
def _test_global_gumbel_softmax(test_case, placement, sbp, tau, dim):
    x = flow.tensor(np.random.randn(20, 32),).to_global(placement=placement, sbp=sbp)
    y_soft = F.gumbel_softmax(x, tau=tau)
    y_hard = F.gumbel_softmax(x, tau=tau, hard=True)
    test_case.assertEqual(x.shape, y_soft.shape)
    test_case.assertEqual(x.shape, y_hard.shape)
    test_case.assertEqual(y_soft.sbp, sbp)
    test_case.assertEqual(y_hard.sbp, sbp)
    test_case.assertEqual(y_soft.placement, placement)
    test_case.assertEqual(y_hard.placement, placement)


@autotest(n=1, check_graph=False)
def _test_global_gumbel_softmax_hard(test_case, placement, sbp, tau, dim):
    x = flow.tensor(np.random.randn(45, 23)).to_global(placement=placement, sbp=sbp)
    y_hard = F.gumbel_softmax(x, tau=tau, dim=dim, hard=True)
    test_case.assertEqual(y_hard.min(), 0)
    if dim == -1:
        test_case.assertEqual(y_hard.sum().item(), 45)
    elif dim == 0:
        test_case.assertEqual(y_hard.sum().item(), 23)


class TestGumbelSoftmaxModule(flow.unittest.TestCase):
    @globaltest
    def test_gumbel_softmax_global(test_case):
        arg_dict = OrderedDict()
        arg_dict["tau"] = [1, 2, 0.5]
        arg_dict["dim"] = [0, -1]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(
                    placement, max_dim=2, except_partial_sum=True
                ):
                    _test_global_gumbel_softmax(test_case, placement, sbp, *arg)
                    _test_global_gumbel_softmax_hard(test_case, placement, sbp, *arg)


if __name__ == "__main__":
    unittest.main()
