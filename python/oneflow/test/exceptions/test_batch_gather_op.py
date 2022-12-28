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
from numpy import array, dtype
import oneflow as flow
import oneflow.unittest


class TestBatchGather(flow.unittest.TestCase):
    def test_input_tensor_dimesion_error_msg(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.tensor(1)
            indice = flow.tensor([1])
            flow.batch_gather(x, indice)
        test_case.assertTrue(
            "The dimension of the input tensor should be greater than zero, but got"
            in str(context.exception)
        )

    def test_indices_dimesion_error_msg(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.tensor([1])
            indice = flow.tensor(1)
            flow.batch_gather(x, indice)
        test_case.assertTrue(
            "The dimension of the indices tensor should be greater than zero, but got"
            in str(context.exception)
        )

    def test_legal_dimension_error_msg(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = np.random.randn(1)
            x_tensor = flow.tensor(x)
            indice = flow.tensor([[1, 1], [1, 1], [1, 1]])
            flow.batch_gather(x_tensor, indice)
        test_case.assertTrue(
            "The dimension of the input tensor should be greater than or equal to the dimension of the indices tensor"
            in str(context.exception)
        )

    def test_indice_type_error_msg(test_case):
        with test_case.assertRaises(TypeError) as context:
            x = np.random.randn(2)
            x_tensor = flow.tensor(x)
            indice = flow.tensor([1, 1], dtype=flow.float64)
            flow.batch_gather(x_tensor, indice)
        test_case.assertTrue(
            "The dtype of the indices tensor must be int32 or int64"
            in str(context.exception)
        )

    def test_tensor_shape_size_error_msg(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = np.random.randn(4, 5)
            x_tensor = flow.tensor(x)
            indice = flow.tensor([[1, 2], [1, 2], [1, 2]])
            out = flow.batch_gather(x_tensor, indice)
        test_case.assertTrue(
            "The size of indices tensor must match the size of input tensor"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
