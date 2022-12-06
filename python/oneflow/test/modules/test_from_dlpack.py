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

import random
import unittest
import os

import torch
import numpy as np

import oneflow as flow
import oneflow.unittest


test_devices = (
    [torch.device("cpu")]
    if os.getenv("ONEFLOW_TEST_CPU_ONLY")
    else [torch.device("cpu"), torch.device("cuda", 0), torch.device("cuda", 1)]
)


def torch_device_to_flow(device):
    if device.type == "cpu":
        return flow.device("cpu")
    elif device.type == "cuda":
        return flow.device("cuda", device.index)
    else:
        raise NotImplementedError("Unsupported device type: {}".format(device.type))


@flow.unittest.skip_unless_1n2d()
class TestFromDlPack(flow.unittest.TestCase):
    def test_same_data(test_case):
        for device in test_devices:
            torch_tensor = torch.randn(3, 4, 5, device=device)
            tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )
            test_case.assertEqual(tensor.size(), (3, 4, 5))
            test_case.assertEqual(tensor.stride(), (20, 5, 1))
            test_case.assertEqual(tensor.storage_offset(), 0)
            test_case.assertEqual(tensor.device, torch_device_to_flow(device))

            tensor[1:2, 2:3, 3:4] = random.random()
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )

    def test_use_ops(test_case):
        for device in test_devices:
            torch_tensor = torch.randn(3, 4, 5, device=device)
            tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            torch_res = torch_tensor ** 2
            res = tensor ** 2
            test_case.assertTrue(np.allclose(res.numpy(), torch_res.cpu().numpy()))

    def test_more_dtype(test_case):
        for device in test_devices:
            for torch_dtype, flow_dtype in [
                (torch.float64, flow.float64),
                (torch.float32, flow.float32),
                (torch.float16, flow.float16),
                (torch.int64, flow.int64),
                (torch.int32, flow.int32),
                (torch.int8, flow.int8),
                (torch.uint8, flow.uint8),
                # PyTorch bfloat16 tensor doesn't support .numpy() method
                # torch.bfloat16,
            ]:
                torch_tensor = torch.ones((2, 3), dtype=torch_dtype, device=device)
                tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
                test_case.assertEqual(tensor.dtype, flow_dtype)
                test_case.assertEqual(tensor.device, torch_device_to_flow(device))
                test_case.assertTrue(
                    np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
                )

    def test_non_contiguous_input(test_case):
        for device in test_devices:
            torch_tensor = torch.randn(2, 3, 4, 5).permute(2, 0, 3, 1).to(device)
            tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            test_case.assertEqual(tensor.device, torch_device_to_flow(device))
            test_case.assertTrue(tensor.shape == torch_tensor.shape)
            test_case.assertTrue(tensor.stride() == torch_tensor.stride())
            test_case.assertTrue(tensor.is_contiguous() == torch_tensor.is_contiguous())
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )

    def test_scalar_tensor(test_case):
        for device in test_devices:
            torch_tensor = torch.tensor(5).to(device)
            tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            test_case.assertEqual(tensor.device, torch_device_to_flow(device))
            test_case.assertTrue(tensor.shape == torch_tensor.shape)
            test_case.assertTrue(tensor.stride() == torch_tensor.stride())
            test_case.assertTrue(tensor.is_contiguous() == torch_tensor.is_contiguous())
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )

    def test_0_size_tensor(test_case):
        for device in test_devices:
            torch_tensor = torch.tensor([]).to(device)
            tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            test_case.assertEqual(tensor.device, torch_device_to_flow(device))
            test_case.assertTrue(tensor.shape == torch_tensor.shape)
            test_case.assertTrue(tensor.stride() == torch_tensor.stride())
            test_case.assertTrue(tensor.is_contiguous() == torch_tensor.is_contiguous())
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )

    def test_lifecycle(test_case):
        for device in test_devices:
            torch_tensor = torch.randn(2, 3, 4, 5).to(device)
            flow_tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            value = flow_tensor.numpy()
            del torch_tensor
            if device.type == "cuda":
                torch.cuda.synchronize()
                # actually release the cuda memory
                torch.cuda.empty_cache()
            test_case.assertTrue(np.array_equal(flow_tensor.numpy(), value))

            torch_tensor = torch.randn(2, 3, 4, 5).to(device)
            flow_tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            value = flow_tensor.numpy()
            del flow_tensor
            if device.type == "cuda":
                flow.cuda.synchronize()
                flow.cuda.empty_cache()
            test_case.assertTrue(np.array_equal(torch_tensor.cpu().numpy(), value))

    def test_subview(test_case):
        for device in test_devices:
            torch_tensor = torch.randn(3, 4, 5, device=device)
            torch_tensor = torch_tensor[1:, :, ::2]
            tensor = flow.from_dlpack(torch.to_dlpack(torch_tensor))
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )
            test_case.assertEqual(tensor.storage_offset(), 0)
            test_case.assertEqual(tensor.stride(), torch_tensor.stride())

            tensor[1:2, ::2, 3:4] = random.random()
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), torch_tensor.cpu().numpy())
            )


if __name__ == "__main__":
    unittest.main()
