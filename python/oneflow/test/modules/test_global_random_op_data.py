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
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


_fn_param = {
    "normal": lambda shape, placement, sbp: flow.normal(
        size=shape, mean=0.0, std=1.0, placement=placement, sbp=sbp
    ),
    "rand": lambda shape, placement, sbp: flow.rand(
        size=shape, placement=placement, sbp=sbp
    ),
    "randint": lambda shape, placement, sbp: flow.randint(
        low=0, high=2, size=shape, placement=placement, sbp=sbp
    ),
    "randn": lambda shape, placement, sbp: flow.randn(
        size=shape, placement=placement, sbp=sbp
    ),
}


def _test_data_consistent(test_case, shape, placement, sbp, fn):
    # lazy result
    class GlobalRandnGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            flow.manual_seed(233)
            x = fn(shape, placement, sbp)
            return x

    model = GlobalRandnGraph()
    lazy_x = model()

    # eager result
    flow.manual_seed(233)
    eager_x = fn(shape, placement, sbp)

    test_case.assertTrue(
        np.array_equal(lazy_x.to_local().numpy(), eager_x.to_local().numpy())
    )

    # different data
    eager_x2 = fn(shape, placement, sbp)

    test_case.assertFalse(
        np.array_equal(eager_x.to_local().numpy(), eager_x2.to_local().numpy())
    )


class TestGlobalRandomOpData(flow.unittest.TestCase):
    @globaltest
    def test_random_op_data_consistent_with_eager_and_lazy(test_case):
        shape = (8, 8)

        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2, except_partial_sum=True):
                for _, fn in _fn_param.items():
                    _test_data_consistent(test_case, shape, placement, sbp, fn=fn)

    @globaltest
    @oneflow.unittest.skip_unless_1n4d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_op_data_correctness(test_case):
        shape = (8, 8)
        sbp = [flow.sbp.split(0), flow.sbp.broadcast]

        for device in ["cpu", "cuda"]:
            placement = flow.placement(device, [[0, 1], [2, 3]])

            for _, fn in _fn_param.items():
                flow.manual_seed(233)
                local_tensor = fn(shape, placement, sbp).to_local().cpu()

                # broadcast local data for each rank
                rank_to_tensor = [
                    local_tensor
                    if rank_id == flow.env.get_rank()
                    else flow.empty(local_tensor.shape, dtype=local_tensor.dtype)
                    for rank_id in range(4)
                ]
                for rank_id in range(4):
                    flow.comm.broadcast(rank_to_tensor[rank_id], rank_id)

                np_local = [x.numpy() for x in rank_to_tensor]
                # rank0 == rank1
                test_case.assertTrue(np.array_equal(np_local[0], np_local[1]))
                # rank2 == rank3
                test_case.assertTrue(np.array_equal(np_local[2], np_local[3]))
                # rank0 != rank2
                test_case.assertFalse(np.array_equal(np_local[0], np_local[2]))
                # rank1 != rank3
                test_case.assertFalse(np.array_equal(np_local[1], np_local[3]))


if __name__ == "__main__":
    unittest.main()
