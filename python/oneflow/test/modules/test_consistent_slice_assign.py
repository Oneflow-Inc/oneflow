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


def _test_logical_slice_assign(test_case, placement, sbp):
    x = random_tensor(2, 4, 4, requires_grad=False).oneflow
    x_numpy = x.detach().cpu().numpy()

    x = x.to_global(placement=placement, sbp=sbp)
    x[:, :2] = 3
    x_numpy[:, :2] = 3

    test_case.assertTrue(x.sbp in [(oneflow.sbp.broadcast,)])
    test_case.assertTrue(np.array_equal(x.numpy(), x_numpy))


def _test_graph_logical_slice_assign(test_case, placement, sbp):
    x = random_tensor(2, 4, 4, requires_grad=False).oneflow
    x_numpy = x.detach().cpu().numpy()

    @flow.nn.Graph.to_graph
    def test_func(x):
        x[:, :2] = 3
        return x

    x = x.to_global(placement=placement, sbp=sbp)

    y = test_func(x)

    x_numpy[:, :2] = 3

    test_case.assertTrue(y.sbp in [(oneflow.sbp.broadcast,)])
    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy))


class TestGlobalLogicalSliceAssign(flow.unittest.TestCase):
    @globaltest
    def test_logical_slice_assign(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                # logical slice assign not support 2d sbp currently
                # logical slice assign only support broadcast currently
                if len(sbp) > 1 or sbp[0] != flow.sbp.broadcast:
                    continue
                _test_logical_slice_assign(test_case, placement, sbp)
                _test_graph_logical_slice_assign(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
