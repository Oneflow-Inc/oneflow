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
class TestTensordotError(flow.unittest.TestCase):
    def test_tensordot_neg_dims_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            a = flow.randn(1, 2, 3)
            b = flow.randn(1, 2, 3)
            flow.tensordot(a, b, dims=-1)
        test_case.assertTrue(
            "tensordot expects dims >= 0, but got dims=-1" in str(context.exception)
        )

    @unittest.skip("PyTorch doesn't have corresponding error message")
    def test_tensordot_too_large_int_dims_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            a = flow.randn(1, 2, 3)
            b = flow.randn(1, 2, 3)
            flow.tensordot(a, b, dims=100)
        test_case.assertTrue(
            "tensordot expects dims <= a.ndim which is 3, but got 100"
            in str(context.exception)
        )

    def test_tensordot_out_of_range_dims_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            a = flow.randn(1, 2, 3)
            b = flow.randn(1, 2, 3)
            flow.tensordot(a, b, dims=[[3], [2]])
        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-3, 2], but got 3)"
            in str(context.exception)
        )

    def test_tensordot_unmatch_dims_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            a = flow.randn(1, 2, 3)
            b = flow.randn(1, 2, 3)
            flow.tensordot(a, b, dims=[[1], [2]])
        test_case.assertTrue(
            "contracted dimensions need to match, but first has size 2 in dim 1 and second has size 3 in dim 2"
            in str(context.exception)
        )

    def test_tensordot_recurring_dim_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            a = flow.randn(1, 2, 3)
            b = flow.randn(1, 2, 3)
            flow.tensordot(a, b, dims=[[1, 1], [1, 1]])
        test_case.assertTrue(
            "dim 1 appears multiple times in the list of dims" in str(context.exception)
        )

    def test_tensordot_dims_different_length_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            a = flow.randn(1, 2, 3)
            b = flow.randn(1, 2, 3)
            flow.tensordot(a, b, dims=[[1], [1, 2]])
        test_case.assertTrue(
            "both dimension lists should have same length" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
