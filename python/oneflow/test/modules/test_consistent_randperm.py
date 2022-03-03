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
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


def _test_consistent_randperm(test_case, N, placement, sbp, dtype):
    x = flow.randperm(N, placement=placement, sbp=sbp, dtype=dtype)

    test_case.assertEqual(x.dtype, dtype)
    # TODO: support (B,S)
    # test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


def _test_graph_randperm(test_case, N, placement, sbp, dtype):
    class ConsistentRandGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = flow.randperm(N, placement=placement, sbp=sbp, dtype=dtype)
            return x

    model = ConsistentRandGraph()
    x = model()

    test_case.assertEqual(x.dtype, dtype)
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


class TestRandConsistent(flow.unittest.TestCase):
    @globaltest
    def test_rand_consistent(test_case):
        RandNs = [i for i in range(10, 50, 5)]
        Dtypes = [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]
        for N in RandNs:
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                    for dtype in Dtypes:
                        _test_consistent_randperm(test_case, N, placement, sbp, dtype)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_rand_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["N"] = [i for i in range(10, 100, 5)]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            flow.placement("cuda", ranks=[[0, 1],]),
        ]
        arg_dict["dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]
        for args in GenArgDict(arg_dict):
            N = args["N"]
            placement = args["placement"]
            dtype = args["dtype"]
            for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                _test_graph_randperm(test_case, N, placement, sbp, dtype)


if __name__ == "__main__":
    unittest.main()
