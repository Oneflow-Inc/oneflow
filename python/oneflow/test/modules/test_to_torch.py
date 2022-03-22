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
import oneflow.unittest
import torch


@flow.unittest.skip_unless_1n1d()
class TestToTroch(flow.unittest.TestCase):
    # NOTE: oneflow and torch cpu tensor shared the same memory, refer to File "python/oneflow/test/modules/test_from_torch.py", line 49, in test_from_torch_cpu.
    def test_to_torch_cpu(test_case):
        flow_t = flow.rand(5, 3, 3)

        torch_t = flow.utils.to_torch(flow_t)

        test_case.assertTrue(
            np.allclose(flow_t.numpy(), torch_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(flow_t.numpy().dtype, torch_t.numpy().dtype)

    def test_to_torch_cpu_with_0_size_data(test_case):
        flow_t = flow.rand(5, 3, 0)

        torch_t = flow.utils.to_torch(flow_t)

        test_case.assertTrue(
            np.allclose(flow_t.numpy(), torch_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(flow_t.numpy().dtype, torch_t.numpy().dtype)

    def test_to_torch_cpu_with_0dim_data(test_case):
        flow_t = flow.tensor(5)

        torch_t = flow.utils.to_torch(flow_t)

        test_case.assertTrue(
            np.allclose(flow_t.numpy(), torch_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(flow_t.numpy().dtype, torch_t.numpy().dtype)

    def _test_to_torch_cuda(test_case):
        # NOTE: This test can not pass, to be fixed later.
        flow_t = flow.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        torch_t = flow.utils.to_torch(flow_t)


if __name__ == "__main__":
    unittest.main()
