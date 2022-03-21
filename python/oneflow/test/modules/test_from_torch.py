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
        torch_t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        numpy_from_torch = torch_t.numpy()

        # NOTE: torch and numpy shared the same memory.
        test_case.assertEqual(
            torch_t.data_ptr(), numpy_from_torch.__array_interface__["data"][0]
        )
        # NOTE: oneflow and numpy shared the same memory,
        #   so oneflow and torch cpu tensor shared the same memory,
        #   which means oneflow can use torch's cpu tensor without cost.
        flow_t = flow.utils.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    def _test_from_torch_cuda(test_case):
        # NOTE: This test can not pass, torch to oneflow conversion of gpu tensor type is not supported.
        #   Because torch does not provide relevant interfaces.
        torch_t = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        flow_t = flow.utils.from_torch(torch_t)


if __name__ == "__main__":
    unittest.main()
