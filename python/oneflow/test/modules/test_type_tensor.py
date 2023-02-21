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

import numpy as np

import oneflow as flow
import oneflow.unittest


type_tensor_all = [
    {
        "cpu_interface": flow.HalfTensor,
        "cuda_interface": flow.cuda.HalfTensor,
        "dtype": flow.float16,
    },
    {
        "cpu_interface": flow.FloatTensor,
        "cuda_interface": flow.cuda.FloatTensor,
        "dtype": flow.float32,
    },
    {
        "cpu_interface": flow.DoubleTensor,
        "cuda_interface": flow.cuda.DoubleTensor,
        "dtype": flow.float64,
    },
    {
        "cpu_interface": flow.BoolTensor,
        "cuda_interface": flow.cuda.BoolTensor,
        "dtype": flow.bool,
    },
    {
        "cpu_interface": flow.ByteTensor,
        "cuda_interface": flow.cuda.ByteTensor,
        "dtype": flow.uint8,
    },
    {
        "cpu_interface": flow.CharTensor,
        "cuda_interface": flow.cuda.CharTensor,
        "dtype": flow.int8,
    },
    {
        "cpu_interface": flow.IntTensor,
        "cuda_interface": flow.cuda.IntTensor,
        "dtype": flow.int32,
    },
    {
        "cpu_interface": flow.LongTensor,
        "cuda_interface": flow.cuda.LongTensor,
        "dtype": flow.int64,
    },
    # TODO: flow.BFloat16Tensor fails to creat Tensor.
    # {"cpu_interface": flow.BFloat16Tensor, "cuda_interface": flow.cuda.BFloat16Tensor, "dtype": flow.bfloat16},
]


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTypeTensor(flow.unittest.TestCase):
    def test_type_tensor(test_case):
        for type_tensor_case in type_tensor_all:
            x = type_tensor_case["cpu_interface"](np.random.randn(2, 3, 4, 5))
            test_case.assertEqual(x.device, flow.device("cpu"))
            test_case.assertEqual(x.dtype, type_tensor_case["dtype"])
            test_case.assertEqual(x.shape, (2, 3, 4, 5))
            test_case.assertFalse(x.requires_grad)
            test_case.assertTrue(x.is_leaf)
            y = type_tensor_case["cuda_interface"](np.random.randn(2, 3, 4, 5))
            test_case.assertEqual(y.device, flow.device("cuda"))
            test_case.assertEqual(y.dtype, type_tensor_case["dtype"])
            test_case.assertEqual(y.shape, (2, 3, 4, 5))
            test_case.assertFalse(y.requires_grad)
            test_case.assertTrue(y.is_leaf)

    def test_doubletensor_corner_cases(test_case):
        corner_cases = [random.randint(1 << 24, 1 << 25) for _ in range(20)]
        test_case.assertTrue(
            np.allclose(
                flow.DoubleTensor(corner_cases).numpy(),
                np.array(corner_cases, dtype=np.float64),
                1e-6,
                1e-6,
            )
        )

    def test_type_tensor_ctor(test_case):
        for tensor_type in type_tensor_all:
            cpu_type = tensor_type["cpu_interface"]
            cuda_type = tensor_type["cuda_interface"]

            # empty ctor
            cpu_type_tensor = cpu_type()
            cuda_type_tensor = cuda_type()
            test_case.assertEqual(cpu_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cpu_type_tensor.device, flow.device("cpu"))
            test_case.assertEqual(cuda_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cuda_type_tensor.device, flow.device("cuda"))

            # other ctor
            other_tensor = flow.Tensor(flow.Size([2, 3, 4, 5]))
            cpu_type_tensor = cpu_type(other_tensor)
            cuda_type_tensor = cuda_type(other_tensor)
            test_case.assertEqual(cpu_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cpu_type_tensor.device, flow.device("cpu"))
            test_case.assertEqual(cuda_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cuda_type_tensor.device, flow.device("cuda"))

            # data ctor
            # numpy inputs have been tested above in test_type_tensor
            data = [random.random() for i in range(20)]
            cpu_type_tensor = cpu_type(data)
            cuda_type_tensor = cuda_type(data)
            test_case.assertEqual(cpu_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cpu_type_tensor.device, flow.device("cpu"))
            test_case.assertEqual(cuda_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cuda_type_tensor.device, flow.device("cuda"))

            # shape ctor
            shape = flow.Size([2, 3, 4, 5])
            cpu_type_tensor = cpu_type(shape)
            cuda_type_tensor = cuda_type(shape)
            test_case.assertEqual(cpu_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cpu_type_tensor.device, flow.device("cpu"))
            test_case.assertEqual(cuda_type_tensor.dtype, tensor_type["dtype"])
            test_case.assertEqual(cuda_type_tensor.device, flow.device("cuda"))


if __name__ == "__main__":
    unittest.main()
