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


def torch_device_to_flow(device):
    if device.type == "cpu":
        return flow.device("cpu")
    elif device.type == "cuda":
        return flow.device("cuda", device.index)
    else:
        raise NotImplementedError("Unsupported device type: {}".format(device.type))


class TestFromTroch(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
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
        flow_t = flow.utils.tensor.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    # NOTE: For the case of 0 size tensor, no memory addresses are compared.
    #  Because the address of 0 size tensor is random at this time.
    @flow.unittest.skip_unless_1n1d()
    def test_from_torch_cpu_with_0_size_data(test_case):
        torch_t = torch.rand(5, 0, 3)

        flow_t = flow.utils.tensor.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    @flow.unittest.skip_unless_1n1d()
    def test_from_torch_cpu_with_0dim_data(test_case):
        torch_t = torch.tensor(5)
        numpy_from_torch = torch_t.numpy()

        test_case.assertEqual(
            torch_t.data_ptr(), numpy_from_torch.__array_interface__["data"][0]
        )

        flow_t = flow.utils.tensor.from_torch(torch_t)

        test_case.assertTrue(
            np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001)
        )
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    @flow.unittest.skip_unless_1n2d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_from_torch_gpu(test_case):
        for device in [torch.device("cuda", 0), torch.device("cuda", 1)]:
            torch_t = torch.tensor([1, 2]).to(device)

            flow_t = flow.utils.tensor.from_torch(torch_t)

            test_case.assertTrue(np.array_equal(torch_t.cpu().numpy(), flow_t.numpy()))
            test_case.assertEqual(torch_t.cpu().numpy().dtype, flow_t.numpy().dtype)
            test_case.assertEqual(torch_device_to_flow(torch_t.device), flow_t.device)

            # Test oneflow tensor and pytorch tensor share the data
            torch_t[0] = 5
            test_case.assertTrue(np.array_equal(torch_t.cpu().numpy(), flow_t.numpy()))


if __name__ == "__main__":
    unittest.main()
