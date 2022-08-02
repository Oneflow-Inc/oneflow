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


class TestSparseSoftmaxCrossEntropyError(flow.unittest.TestCase):
    def test_sparse_softmax_cross_entropy_prediction_numaxes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            prediction = flow.randn(10)
            label = flow.randint(0, 10, (10, 10), dtype=flow.int64)
            flow._C.sparse_softmax_cross_entropy(prediction, label)
        test_case.assertTrue(
            "The dimension of prediction must be greater than or equal to 2, but found"
            in str(context.exception)
        )

    def test_sparse_softmax_cross_entropy_label_numaxes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            prediction = flow.randn(10, 10, 5)
            label = flow.randint(0, 10, (10, 10, 5), dtype=flow.int64)
            flow._C.sparse_softmax_cross_entropy(prediction, label)
        test_case.assertTrue(
            "The dimension of label is expected to be less than that of prediction by 1"
            in str(context.exception)
        )

    def test_sparse_softmax_cross_entropy_prediction_i_shape_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            prediction = flow.randn(10, 10, 5)
            label = flow.randint(0, 10, (10, 9), dtype=flow.int64)
            flow._C.sparse_softmax_cross_entropy(prediction, label)
        test_case.assertTrue("must match the size of label" in str(context.exception))

    def test_sparse_softmax_cross_entropy_label_dtype_err(test_case):
        with test_case.assertRaises(TypeError) as context:
            prediction = flow.randn(10, 10, 5)
            label = flow.randn(10, 10, dtype=flow.float32)
            flow._C.sparse_softmax_cross_entropy(prediction, label)
        test_case.assertTrue(
            "The dtype of label must be integer, but found " in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
