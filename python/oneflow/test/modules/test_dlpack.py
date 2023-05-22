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
from oneflow.test_utils.test_util import GenCartesianProduct


test_device_args = (
    [("cpu",)]
    if os.getenv("ONEFLOW_TEST_CPU_ONLY")
    else [("cpu",), ("cuda", 0), ("cuda", 1)]
)
test_args = list(
    GenCartesianProduct((test_device_args, [(torch, flow), (flow, torch)]))
)


def are_tensors_equal(a, b):
    def are_devices_equal(a, b):
        if a.type == "cuda" and b.type == "cuda":
            return a.index == b.index
        else:
            return a.type == b.type

    return (
        np.array_equal(a.cpu().numpy(), b.cpu().numpy())
        and are_devices_equal(a.device, b.device)
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.cpu().numpy().dtype == b.cpu().numpy().dtype
    )


@flow.unittest.skip_unless_1n2d()
class TestPack(flow.unittest.TestCase):
    def test_same_data(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = m1.randn(3, 4, 5, device=m1.device(*device_args))
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))
            test_case.assertEqual(tensor2.storage_offset(), 0)

            tensor2[1:2, 2:3, 3:4] = random.random()
            # NOTE: OneFlow operations are asynchoronously executed,
            # so we need to synchronize explicitly here.
            flow._oneflow_internal.eager.Sync()
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))

    def test_use_ops(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = m1.randn(3, 4, 5, device=m1.device(*device_args))
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            res1 = tensor1 ** 2
            res2 = tensor2 ** 2
            test_case.assertTrue(np.allclose(res1.cpu().numpy(), res2.cpu().numpy()))

    def test_more_dtype(test_case):
        # PyTorch bfloat16 tensor doesn't support .numpy() method
        # so we can't test it
        # torch.bfloat16, flow.bfloat16
        dtypes = ["float64", "float32", "float16", "int64", "int32", "int8", "uint8"]

        for device_args, (m1, m2) in test_args:
            for dtype in dtypes:
                tensor1 = m1.ones(
                    (2, 3), dtype=getattr(m1, dtype), device=m1.device(*device_args)
                )
                tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
                test_case.assertTrue(are_tensors_equal(tensor1, tensor2))

    def test_non_contiguous_input(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = (
                m1.randn(2, 3, 4, 5).permute(2, 0, 3, 1).to(m1.device(*device_args))
            )
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))

    def test_scalar_tensor(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = m1.tensor(5).to(m1.device(*device_args))
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))

    def test_0_size_tensor(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = m1.tensor([]).to(m1.device(*device_args))
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))

    def test_lifecycle(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = m1.randn(2, 3, 4, 5).to(m1.device(*device_args))
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            value = tensor1.cpu().numpy()
            del tensor2
            if device_args[0] == "cuda":
                m2.cuda.synchronize()
                # actually release the cuda memory
                m2.cuda.empty_cache()
            test_case.assertTrue(np.array_equal(tensor1.cpu().numpy(), value))

            tensor1 = m1.randn(2, 3, 4, 5).to(m1.device(*device_args))
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            value = tensor2.cpu().numpy()
            del tensor1
            if device_args[0] == "cuda":
                m1.cuda.synchronize()
                m1.cuda.empty_cache()
            test_case.assertTrue(np.array_equal(tensor2.cpu().numpy(), value))

    def test_subview(test_case):
        for device_args, (m1, m2) in test_args:
            tensor1 = m1.randn(3, 4, 5, device=m1.device(*device_args))
            tensor1 = tensor1[1:, :, ::2]
            tensor2 = m2.from_dlpack(m1.to_dlpack(tensor1))
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))
            test_case.assertEqual(tensor2.storage_offset(), 0)

            tensor2[1:2, ::2, 3:4] = random.random()
            test_case.assertTrue(are_tensors_equal(tensor1, tensor2))


if __name__ == "__main__":
    unittest.main()
