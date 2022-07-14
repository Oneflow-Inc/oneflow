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


def _test_consistent_randint_like(test_case, shape, placement, sbp, dtype):
    x_ = flow.randint(1, 10, shape)
    x = flow.randint_like(x_, 1, 10, placement=placement, sbp=sbp, dtype=dtype)

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)
    test_case.assertEqual(x.dtype, dtype)


def _test_graph_randint_like(test_case, shape, placement, sbp, dtype):
    class ConsistentRandIntLikeGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x_ = flow.randint(1, 10, shape)
            x = flow.randint_like(x_, 1, 10, placement=placement, sbp=sbp, dtype=dtype)
            return x

    model = ConsistentRandIntLikeGraph()
    x = model()

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)
    test_case.assertEqual(x.dtype, dtype)


class TestRandIntLikeConsistent(flow.unittest.TestCase):
    @globaltest
    def test_randint_like_consistent(test_case):
        shapes = [(8,), (8, 8,), (8, 8, 8)]
        dtypes = [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]
        for shape in shapes:
            for placement in all_placement():
                for sbp in all_sbp(
                    placement, max_dim=len(shape), except_partial_sum=True
                ):
                    for dtype in dtypes:
                        _test_consistent_randint_like(
                            test_case, shape, placement, sbp, dtype
                        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_randint_like_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(8,), (8, 8,), (8, 8, 8)]
        arg_dict["dtype"] = [
            flow.uint8,
            flow.int32,
            flow.float32,
        ]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            flow.placement("cuda", ranks=[[0, 1],]),
        ]
        for args in GenArgDict(arg_dict):
            shape = args["shape"]
            placement = args["placement"]
            dtype = args["dtype"]
            for sbp in all_sbp(placement, max_dim=len(shape), except_partial_sum=True):
                _test_graph_randint_like(test_case, shape, placement, sbp, dtype)


if __name__ == "__main__":
    unittest.main()
