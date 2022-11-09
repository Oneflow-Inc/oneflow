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
import oneflow.nn.functional as F
import torch


@flow.unittest.skip_unless_1n1d()
class TestRepeatInterleave(flow.unittest.TestCase):
    def test_repeat_interleave_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1, 2], [3, 4]])
            y = flow.repeat_interleave(x, 3, dim=4)
        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-2, 1], but got 4)"
            in str(context.exception)
        )

    def test_repeat_interleave_tensor_shape_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1, 2], [3, 4]])
            r = flow.tensor([[1, 2], [3, 4]])
            y = flow.repeat_interleave(x, r, dim=1)
        test_case.assertTrue(
            "repeat_interleave only accept 1D vector as repeat"
            in str(context.exception)
        )

    def test_repeat_interleave_dtype_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1, 2], [3, 4]])
            r = flow.tensor([1.0, 2.0])
            y = flow.repeat_interleave(x, r, dim=1)
        test_case.assertTrue("repeats has to be Long tensor" in str(context.exception))

    def test_repeat_interleave_negative_tensor_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1, 2], [3, 4]])
            r = flow.tensor([1, -2])
            y = flow.repeat_interleave(x, r, dim=1)
        test_case.assertTrue("repeats can not be negative" in str(context.exception))

    def test_repeat_interleave_negative_tensor_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1, 2], [3, 4]])
            r = flow.tensor([1, 2])
            y = flow.repeat_interleave(x, r, dim=2)
        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-2, 1], but got 2)"
            in str(context.exception)
        )

    def test_repeat_interleave_dim_not_match_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1, 2], [3, 4]])
            r = flow.tensor([1])
            y = flow.repeat_interleave(x, r, dim=1)
        test_case.assertTrue(
            "repeats must have the same size as input along dim"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
