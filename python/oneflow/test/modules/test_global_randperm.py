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
import numpy as np
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


def _test_global_randperm(test_case, N, placement, sbp, dtype):
    x = flow.randperm(N, placement=placement, sbp=sbp, dtype=dtype)
    # TODO:Synchronously get a global random seed, and then each rank sets its own seed in manual_seeds
    test_case.assertEqual(x.dtype, dtype)
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


def _test_graph_randperm(test_case, N, placement, sbp, dtype):
    class GlobalRandpermGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = flow.randperm(N, placement=placement, sbp=sbp, dtype=dtype)
            return x

    model = GlobalRandpermGraph()
    x = model()
    y1 = x.to_global(placement=placement, sbp=sbp)
    y1_np_sort = np.sort(y1.numpy())
    y2 = np.arange(N)
    test_case.assertTrue(np.allclose(y1_np_sort, y2, atol=1e-4, rtol=1e-4))
    test_case.assertEqual(x.dtype, dtype)
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


@unittest.skip("This fails in multi-gpu")
class TestRandpermGlobal(flow.unittest.TestCase):
    @globaltest
    def test_randperm_global(test_case):
        RandNs = [i for i in range(10, 50, 10)]
        # TODO support uint8,int8,int64,float32,float64,data type test
        Dtypes = [
            flow.int32,
        ]
        for N in RandNs:
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                    for dtype in Dtypes:
                        _test_global_randperm(test_case, N, placement, sbp, dtype)

    @flow.unittest.skip_unless_1n2d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @globaltest
    def test_randperm_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["N"] = [i for i in range(10, 50, 10)]
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
