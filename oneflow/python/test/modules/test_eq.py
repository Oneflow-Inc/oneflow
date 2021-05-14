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
class TestEq(flow.unittest.TestCase):
    def test_eq(test_case):
        arr1 = np.array([2, 3, 4, 5,])
        arr2 = np.array([2, 3, 4, 1])
        input = flow.Tensor(arr1, dtype=flow.float32)
        other = flow.Tensor(arr2, dtype=flow.float32)

        of_out = flow.eq(input, other)
        of_out2 = flow.equal(input, other)
        np_out = np.equal(arr1, arr2)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
        test_case.assertTrue(np.array_equal(of_out2.numpy(), np_out))

    def test_eq_tensor_function(test_case):
        arr1 = np.random.randint(1, 10, size=(2, 3, 4, 5))
        arr2 = np.random.randint(1, 10, size=(2, 3, 4, 5))
        input = flow.Tensor(arr1, dtype=flow.float32)
        other = flow.Tensor(arr2, dtype=flow.float32)

        of_out = input.eq(other)
        np_out = np.equal(arr1, arr2)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
