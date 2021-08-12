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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest


class TestConsistentCast(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cpu_local_tensor_to_gpu_placement(test_case):
        if flow.distributed.get_rank() == 0:
            np_arr = np.array([4, 6, 7, 8], dtype=np.float32)
        else:
            np_arr = np.array([0, 0, 0, 0], dtype=np.float32)
        tensor = flow.Tensor(np_arr, dtype=flow.float32)
        placement = flow.placement("cuda", {0: range(4)})
        device = flow.device("cuda")
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(consistent_tensor.to_local().device == device)
        test_case.assertTrue(consistent_tensor.placement == placement)
        test_case.assertTrue(
            np.array_equal(
                consistent_tensor.to_local().numpy(),
                np.array([4, 6, 7, 8], dtype=np.float32),
            )
        )

    @flow.unittest.skip_unless_1n4d()
    def test_cpu_p2b_size_16(test_case):
        tensor = flow.ones((16,), dtype=flow.float32)
        placement = flow.placement("cpu", {0: range(4)})
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.partial_sum)
        consistent_tensor = consistent_tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(
            np.array_equal(
                consistent_tensor.to_local().numpy(),
                np.ones((16,), dtype=np.float32) * 4,
            )
        )

    @flow.unittest.skip_unless_1n4d()
    def test_cpu_p2b_size_3(test_case):
        tensor = flow.ones((3,), dtype=flow.float32)
        placement = flow.placement("cpu", {0: range(4)})
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.partial_sum)
        consistent_tensor = consistent_tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(
            np.array_equal(
                consistent_tensor.to_local().numpy(),
                np.ones((3,), dtype=np.float32) * 4,
            )
        )

    @flow.unittest.skip_unless_1n4d()
    def test_cpu_p2b_size_1(test_case):
        tensor = flow.ones((1,), dtype=flow.float32)
        placement = flow.placement("cpu", {0: range(4)})
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.partial_sum)
        consistent_tensor = consistent_tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(
            np.array_equal(
                consistent_tensor.to_local().numpy(),
                np.ones((1,), dtype=np.float32) * 4,
            )
        )

    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_local_to_consistent_with_wrong_device(test_case):
        np_arr = np.array([4, 6], dtype=np.float32)
        tensor = flow.Tensor(
            np_arr,
            device=flow.device("cuda:%d" % ((flow.distributed.get_rank() + 1) % 4)),
            dtype=flow.float32,
        )
        placement = flow.placement("cuda", {0: range(4)})
        device = flow.device("cuda")
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(consistent_tensor.to_local().device == device)
        test_case.assertTrue(consistent_tensor.placement == placement)


if __name__ == "__main__":
    unittest.main()
