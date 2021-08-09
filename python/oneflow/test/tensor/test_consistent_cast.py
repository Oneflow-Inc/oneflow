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
    @flow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cpu_local_tensor_to_gpu_placement(test_case):
        np_arr = np.array([4, 6], dtype=np.float32)
        tensor = flow.Tensor(np_arr, dtype=flow.float32)
        placement = flow.placement("cuda", {0: [0]})
        device = flow.device("cuda")
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(consistent_tensor.to_local().device == device)
        test_case.assertTrue(consistent_tensor.placement == placement)

    @flow.unittest.skip_unless_1n2d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_local_to_consistent_with_wrong_device(test_case):
        np_arr = np.array([4, 6], dtype=np.float32)
        tensor = flow.Tensor(np_arr, dtype=flow.float32)
        tensor.to("cuda:1")
        placement = flow.placement("cuda", {0: [0]})
        device = flow.device("cuda")
        consistent_tensor = tensor.to_consistent(placement, flow.sbp.broadcast)
        test_case.assertTrue(consistent_tensor.to_local().device == device)
        test_case.assertTrue(consistent_tensor.placement == placement)


if __name__ == "__main__":
    unittest.main()
