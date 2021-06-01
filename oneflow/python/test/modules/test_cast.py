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
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCast(flow.unittest.TestCase):
    def test_cast_float2int(test_case):
        np_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        input = flow.Tensor(np_arr, dtype=flow.float32)
        output = flow.cast(input, flow.int8)
        np_out = np_arr.astype(np.int8)
        test_case.assertTrue(np.array_equal(output.numpy(), np_out))

    def test_cast_int2float(test_case):
        np_arr = np.random.randn(5, 2, 3, 4).astype(np.int8)
        input = flow.Tensor(np_arr, dtype=flow.int8)
        output = flow.cast(input, flow.float32)
        np_out = np_arr.astype(np.float32)
        test_case.assertTrue(np.array_equal(output.numpy(), np_out))

    def test_cast_tensor_function(test_case):
        np_arr = np.random.randn(1, 2, 3, 4).astype(np.float32)
        input = flow.Tensor(np_arr, dtype=flow.float32)
        output = input.cast(flow.int8)
        np_out = np_arr.astype(np.int8)
        test_case.assertTrue(np.array_equal(output.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
