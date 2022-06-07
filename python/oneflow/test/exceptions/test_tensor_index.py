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
import oneflow.unittest
import oneflow as flow


class TestTensorIndexError(flow.unittest.TestCase):
    def test_PrepareSliceIndices_indices_amount_index_error(test_case):
        with test_case.assertRaises(IndexError) as context:
            x = flow.arange(16).reshape(4, 4)
            x[0, 0, 0] = 0
        test_case.assertTrue(
            "Too many indices for tensor of dimension" in str(context.exception)
        )

    def test_PrepareSliceIndices_slice_step_runtime_error(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.tensor([0, 1, 2, 3], dtype=flow.int32)
            s = slice(0, 2, -1)
            y = x[s]
        test_case.assertTrue("Step must be greater than zero" in str(context.exception))

    def test_ApplySelectIndexing_input_dim_runtime_error(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.tensor(5, dtype=flow.int32)
            y = x[0]
        test_case.assertTrue(
            "select() cannot be applied to a 0-dim tensor." in str(context.exception)
        )

    def test_ApplySelectIndexing_index_error(test_case):
        with test_case.assertRaises(IndexError) as context:
            x = flow.ones(2, 3, dtype=flow.int32)
            y = x[3]
        test_case.assertTrue(
            "Index out of range (expected to be in range of" in str(context.exception)
        )

    def test_ApplyAdvancedIndexing_index_error(test_case):
        with test_case.assertRaises(IndexError) as context:
            x = flow.ones(2, 2, dtype=flow.int32)
            index = (
                flow.tensor(1, dtype=flow.int32),
                flow.tensor(1, dtype=flow.int32),
                flow.tensor(1, dtype=flow.int32),
            )
            y = x[index]
        test_case.assertTrue(
            "Too many indices for tensor of dimension" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
