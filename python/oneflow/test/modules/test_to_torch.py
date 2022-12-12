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
import os

import oneflow as flow
import oneflow.unittest
import torch


@flow.unittest.skip_unless_1n1d()
class TestToTroch(flow.unittest.TestCase):
    # NOTE: oneflow and torch cpu tensor shared the same memory, refer to File "python/oneflow/test/modules/test_from_torch.py", line 49, in test_from_torch_cpu.
    def test_to_torch_cpu(test_case):
        flow_t = flow.rand(5, 3, 3)
        numpy_from_flow = flow_t.numpy()

        torch_t = flow.utils.tensor.to_torch(flow_t)

        test_case.assertEqual(
            torch_t.data_ptr(), numpy_from_flow.__array_interface__["data"][0]
        )
        numpy_from_flow[0][0] = [1, 2, 3]
        test_case.assertTrue(
            np.allclose(torch_t.numpy(), numpy_from_flow, rtol=0.001, atol=0.001)
        )

        test_case.assertTrue(
            np.allclose(flow_t.numpy(), torch_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(flow_t.numpy().dtype, torch_t.numpy().dtype)

    # NOTE: For the case of 0 size tensor, no memory addresses are compared.
    #  Because the address of 0 size tensor is random at this time.
    def test_to_torch_cpu_with_0_size_data(test_case):
        flow_t = flow.rand(5, 3, 0)

        torch_t = flow.utils.tensor.to_torch(flow_t)

        test_case.assertTrue(
            np.allclose(flow_t.numpy(), torch_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(flow_t.numpy().dtype, torch_t.numpy().dtype)

    def test_to_torch_cpu_with_0dim_data(test_case):
        flow_t = flow.tensor(5)
        numpy_from_flow = flow_t.numpy()

        torch_t = flow.utils.tensor.to_torch(flow_t)

        test_case.assertEqual(
            torch_t.data_ptr(), numpy_from_flow.__array_interface__["data"][0]
        )

        test_case.assertTrue(
            np.allclose(flow_t.numpy(), torch_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(flow_t.numpy().dtype, torch_t.numpy().dtype)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_to_torch_gpu(test_case):
        flow_t = flow.rand(5, 3, 3).to("cuda")

        torch_t = flow.utils.tensor.to_torch(flow_t)

        flow_t[0][0] = flow.tensor([1, 2, 3]).to(flow.float32)
        # NOTE: OneFlow operations are asynchoronously executed,
        # so we need to synchronize explicitly here.
        flow._oneflow_internal.eager.Sync()
        test_case.assertTrue(np.array_equal(torch_t.cpu().numpy(), flow_t.numpy()))

        test_case.assertEqual(flow_t.numpy().dtype, torch_t.cpu().numpy().dtype)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_to_torch_global(test_case):
        flow_t = flow.rand(5, 3, 3).to_global(
            placement=flow.placement.all("cuda"), sbp=flow.sbp.broadcast
        )

        torch_t = flow.utils.tensor.to_torch(flow_t)

        test_case.assertEqual(flow_t.numpy().dtype, torch_t.cpu().numpy().dtype)


if __name__ == "__main__":
    unittest.main()
