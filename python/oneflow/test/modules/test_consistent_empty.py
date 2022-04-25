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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


def _test_consistent_empty(test_case, func, shape, placement, sbp):
    func2 = None
    if func == "empty":
        func = flow.empty
    elif func == "new_empty":
        func = flow.empty
        func2 = flow.new_empty
    else:
        raise NotImplementedError

    x = func(*shape, placement=placement, sbp=sbp)
    if func2:
        x = func2(x, size=shape)

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


def _test_graph_empty(test_case, func, shape, placement, sbp):
    func2 = None
    if func == "empty":
        func = flow.empty
    elif func == "new_empty":
        func = flow.empty
        func2 = flow.new_empty
    else:
        raise NotImplementedError

    class ConsistentEmptyGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = func(*shape, placement=placement, sbp=sbp)
            if func2:
                x = func2(x, size=shape)
            return x

    model = ConsistentEmptyGraph()
    x = model()

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


class TestEmptyConsistent(flow.unittest.TestCase):
    @globaltest
    def test_empty_consistent(test_case):
        shapes = [(8,), (8, 8,), (8, 8, 8)]
        functions = [
            "empty",
            "new_empty",
        ]
        for func in functions:
            for shape in shapes:
                for placement in all_placement():
                    for sbp in all_sbp(
                        placement, max_dim=len(shape), except_partial_sum=True
                    ):
                        _test_consistent_empty(test_case, func, shape, placement, sbp)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_empty_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["func"] = ["empty", "new_empty"]
        arg_dict["shape"] = [(8,), (8, 8,), (8, 8, 8)]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            flow.placement("cuda", ranks=[[0, 1],]),
        ]
        for args in GenArgDict(arg_dict):
            func = args["func"]
            shape = args["shape"]
            placement = args["placement"]
            for sbp in all_sbp(placement, max_dim=len(shape), except_partial_sum=True):
                _test_graph_empty(test_case, func, shape, placement, sbp)


if __name__ == "__main__":
    unittest.main()
