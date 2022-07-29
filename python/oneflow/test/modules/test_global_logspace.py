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

from collections import OrderedDict

import unittest

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgDict


def _test_global_logspace(test_case, placement, sbp):
    x = flow.logspace(start=-10, end=10, steps=8, placement=placement, sbp=sbp)

    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


def _test_graph_logspace(test_case, start, end, steps, placement, sbp):
    class GlobalLogspaceGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = flow.logspace(start, end, steps, placement=placement, sbp=sbp)
            return x

    model = GlobalLogspaceGraph()
    x = model()

    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


class TestLogspaceGlobal(flow.unittest.TestCase):
    # TODO(wyg): It will be infer all broadcast sbp when 1n1d,
    #            slice_update will get error when doing inplace operator.
    #            Remove this judgement after refactor sbp infer method in Operator class.
    @globaltest
    def test_logspace_global(test_case):
        for placement in all_placement():
            if placement.ranks.size == 1:
                continue
            for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                _test_global_logspace(test_case, placement, sbp)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_logspace_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["start"] = [-2, 0, 2]
        arg_dict["end"] = [4, 8, 16]
        arg_dict["steps"] = [8, 16, 24]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            flow.placement("cuda", ranks=[[0, 1],]),
        ]
        for args in GenArgDict(arg_dict):
            start = args["start"]
            end = args["end"]
            steps = args["steps"]
            placement = args["placement"]
            for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                _test_graph_logspace(test_case, start, end, steps, placement, sbp)


if __name__ == "__main__":
    unittest.main()
