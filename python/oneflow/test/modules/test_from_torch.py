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
class TestFromTroch(flow.unittest.TestCase):
    def test_from_torch_cpu(test_case):
        torch_t = torch.rand(5, 3, 3)
        numpy_from_torch = torch_t.numpy()

        # NOTE: torch and numpy shared the same memory.
        test_case.assertEqual(
            torch_t.data_ptr(), numpy_from_torch.__array_interface__["data"][0]
        )
        numpy_from_torch[0][0] = [1, 2, 3]
        test_case.assertTrue(
            np.allclose(torch_t.numpy(), numpy_from_torch, rtol=0.001, atol=0.001)
        )

        # NOTE: oneflow and numpy shared the same memory,
        #   so oneflow and torch cpu tensor shared the same memory,
        #   which means oneflow can use torch's cpu tensor without cost.
        flow_t = flow.utils.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    # NOTE: For the case of 0 size tensor, no memory addresses are compared.
    #  Because the address of 0 size tensor is random at this time.
    def test_from_torch_cpu_with_0_size_data(test_case):
        torch_t = torch.rand(5, 0, 3)

        flow_t = flow.utils.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    def test_from_torch_cpu_with_0dim_data(test_case):
        torch_t = torch.tensor(5)
        numpy_from_torch = torch_t.numpy()

        test_case.assertEqual(
            torch_t.data_ptr(), numpy_from_torch.__array_interface__["data"][0]
        )

        flow_t = flow.utils.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)


if __name__ == "__main__":
    unittest.main()
