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


class TestNewTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_new_tensor_local_mode_with_default_args(test_case):
        tensor = flow.randn(5)
        data = [[1, 2], [3, 4]]
        new_tensor = tensor.new_tensor(data)
        test_case.assertEqual(new_tensor.dtype, tensor.dtype)
        test_case.assertEqual(new_tensor.device, tensor.device)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n1d()
    def test_new_tensor_local_mode_with_spec_args(test_case):
        tensor = flow.randn(5)
        data = [[1, 2], [3, 4]]
        new_tensor = tensor.new_tensor(data, flow.int64, "cuda")
        test_case.assertEqual(new_tensor.dtype, flow.int64)
        test_case.assertEqual(new_tensor.device, flow.device("cuda"))

    @flow.unittest.skip_unless_1n2d()
    def test_new_tensor_global_mode_with_default_args(test_case):
        placement = flow.placement(type="cpu", ranks=[0, 1])
        sbp = flow.sbp.split(0)
        tensor = flow.randn(4, 4, placement=placement, sbp=sbp)
        data = [[1, 2], [3, 4]]
        new_tensor = tensor.new_tensor(data)
        test_case.assertEqual(new_tensor.dtype, tensor.dtype)
        test_case.assertEqual(new_tensor.placement, placement)
        test_case.assertEqual(new_tensor.sbp, (sbp,))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_new_tensor_global_mode_with_spec_args(test_case):
        placement = flow.placement(type="cuda", ranks=[0, 1])
        sbp = flow.sbp.split(0)
        tensor = flow.randn(4, 4, placement=placement, sbp=sbp)
        data = [[1, 2], [3, 4]]
        new_tensor = tensor.new_tensor(
            data, placement=placement, sbp=flow.sbp.broadcast
        )
        test_case.assertEqual(new_tensor.dtype, tensor.dtype)
        test_case.assertEqual(new_tensor.placement, placement)
        test_case.assertEqual(new_tensor.sbp, (flow.sbp.broadcast,))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n1d()
    def test_new_cuda_bfloat16_local_tensor_with_numpy(test_case):
        from oneflow import sysconfig

        if sysconfig.get_cuda_version() < 11000:
            return
        np_array = np.random.rand(4, 4)
        tensor = flow.tensor(np_array, dtype=flow.bfloat16, device="cuda")
        test_case.assertEqual(tensor.dtype, flow.bfloat16)
        test_case.assertEqual(tensor.device, flow.device("cuda"))

    @flow.unittest.skip_unless_1n1d()
    def test_new_cpu_bfloat16_local_tensor_with_numpy(test_case):
        np_array = np.random.rand(4, 4)
        tensor = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        test_case.assertEqual(tensor.dtype, flow.bfloat16)
        test_case.assertEqual(tensor.device, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
