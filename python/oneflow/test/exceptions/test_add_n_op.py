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
import oneflow as flow
import oneflow.unittest


class TestAddN(flow.unittest.TestCase):
    def test_add_n_shape_error_msg(test_case):
        a = flow.tensor([1, 2])
        b = flow.tensor([3, 4])
        c = flow.tensor([[2, 2], [2, 2]])
        with test_case.assertRaises(RuntimeError) as context:
            flow.add(a, b, c)
        test_case.assertTrue(
            "inconsistent tensor size, expected all tensor to have the same number of elements, but got"
            in str(context.exception)
        )

    def test_add_n_dtype_error_msg(test_case):
        a = flow.tensor([1, 2], dtype=flow.int64)
        b = flow.tensor([3, 4], dtype=flow.int64)
        c = flow.tensor([2, 2], dtype=flow.float64)
        with test_case.assertRaises(RuntimeError) as context:
            flow.add(a, b, c)
        test_case.assertTrue(
            "expected all tenser to have same type, but found" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
