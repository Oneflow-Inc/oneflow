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

import torch
import numpy as np

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestFromNumpy(flow.unittest.TestCase):
    def test_same_data(test_case):
        np_arr = np.random.randn(3, 4, 5)
        tensor = flow.from_numpy(np_arr)
        test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))
        test_case.assertEqual(tensor.size(), (3, 4, 5))
        test_case.assertEqual(tensor.stride(), (20, 5, 1))
        test_case.assertEqual(tensor.storage_offset(), 0)

        np_arr[1:2, 2:3, 3:4] = random.random()
        test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))

    def test_use_ops(test_case):
        np_arr = np.random.randn(3, 4, 5)
        tensor = flow.from_numpy(np_arr)
        res = tensor ** 2
        test_case.assertTrue(np.allclose(np_arr ** 2, res.numpy()))

    def test_more_dtype(test_case):
        for dtype in [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int8,
            np.uint8,
        ]:
            np_arr = np.ones((2, 3), dtype=dtype)
            tensor = flow.from_numpy(np_arr)
            # TODO(wyg): oneflow.float16 do not support to copy from tensor to numpy
            if tensor.dtype not in [flow.float16]:
                test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))

    def test_non_contiguous_input(test_case):
        np_arr = np.random.randn(2, 3, 4, 5).transpose(2, 0, 3, 1)
        flow_tensor = flow.from_numpy(np_arr)
        torch_tensor = torch.from_numpy(np_arr)
        test_case.assertTrue(flow_tensor.shape == torch_tensor.shape)
        test_case.assertTrue(flow_tensor.stride() == torch_tensor.stride())
        test_case.assertTrue(
            flow_tensor.is_contiguous() == torch_tensor.is_contiguous()
        )
        test_case.assertTrue(np.array_equal(flow_tensor.numpy(), torch_tensor.numpy()))


if __name__ == "__main__":
    unittest.main()
