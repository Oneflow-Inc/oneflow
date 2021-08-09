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

import numpy as np

import oneflow as flow
import oneflow.typing as oft
import oneflow.unittest


class TestBoxing(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_boxing_single_device(test_case):
        np_arr = np.array([1, 2, 3])
        x = flow.tensor(np_arr).to("cuda")
        x = x.to_consistent(flow.placement("cuda", {0: range(1)}), flow.sbp.split(0))
        x = x.to_consistent(sbp=flow.sbp.broadcast)
        test_case.assertTrue(np.array_equal(np_arr, x.to_local().numpy()))

    @flow.unittest.skip_unless_1n2d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_boxing_two_devices(test_case):
        rank = oneflow.framework.distribute.get_rank()
        np_arr = np.array([1, 2, 3]) * (rank + 1)
        x = flow.tensor(np_arr).to("cuda")

        def assert_broadcast(x):
            test_case.assertTrue(x.sbp[0] == flow.sbp.broadcast)
            test_case.assertTrue(
                np.array_equal(np.array([1, 2, 3, 2, 4, 6]), x.to_local().numpy())
            )

        def assert_partial_sum(x):
            test_case.assertTrue(x.sbp[0] == flow.sbp.partial_sum)
            if rank == 0:
                test_case.assertTrue(
                    np.array_equal(np.array([1, 2, 3, 2, 4, 6]), x.to_local().numpy())
                )
            else:
                test_case.assertTrue(
                    np.array_equal(np.zeros((6,)), x.to_local().numpy())
                )

        def assert_split(x):
            test_case.assertTrue(x.sbp[0] == flow.sbp.split(0))
            if rank == 0:
                test_case.assertTrue(
                    np.array_equal(np.array([1, 2, 3]), x.to_local().numpy())
                )
            else:
                test_case.assertTrue(
                    np.array_equal(np.array([2, 4, 6]), x.to_local().numpy())
                )

        x = x.to_consistent(flow.placement("cuda", {0: range(2)}), flow.sbp.split(0))

        # S -> B
        x = x.to_consistent(sbp=flow.sbp.broadcast)
        assert_broadcast(x)

        # B -> P
        x = x.to_consistent(sbp=flow.sbp.partial_sum)
        assert_partial_sum(x)

        # P -> S
        x = x.to_consistent(sbp=flow.sbp.split(0))
        assert_split(x)

        # S -> P
        x = x.to_consistent(sbp=flow.sbp.partial_sum)
        assert_partial_sum(x)

        # P -> B
        x = x.to_consistent(sbp=flow.sbp.broadcast)
        assert_broadcast(x)

        # B -> S
        x = x.to_consistent(sbp=flow.sbp.split(0))
        assert_split(x)


if __name__ == "__main__":
    unittest.main()
