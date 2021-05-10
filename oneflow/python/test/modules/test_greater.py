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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGreater(flow.unittest.TestCase):
    def test_greater_v1(test_case):
        input1 = flow.Tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)
        input2 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        of_out = flow.gt(input1, input2)
        np_out = np.greater(input1.numpy(), input2.numpy())
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    def test_tensor_greater(test_case):
        input1 = flow.Tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)
        input2 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        of_out = input1.gt(input2)
        np_out = np.greater(input1.numpy(), input2.numpy())
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    def test_greater_symbol(test_case):
        input1 = flow.Tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)
        input2 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        of_out = input1 > input2
        np_out = np.greater(input1.numpy(), input2.numpy())
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    
    def test_greater_int_scalar(test_case):
        np_arr = np.random.randn(2, 3, 4, 5)
        input1 = flow.Tensor(np_arr, dtype=flow.float32)
        input2 = 1
        of_out = input1 > input2
        np_out = np.greater(np_arr, input2)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    
    def test_greater_float_scalar(test_case):
        np_arr = np.random.randn(3, 2, 5, 7)
        input1 = flow.Tensor(np_arr, dtype=flow.float32)
        input2 = 2.3
        of_out = input1 > input2
        np_out = np.greater(np_arr, input2)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
