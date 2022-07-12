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


class TestSoftmaxCrossEntropyError(flow.unittest.TestCase):
    def test_softmax_cross_entropy_prediction_numaxes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            prediction = flow.randn(10)
            label = flow.randn(1, 10)
            flow._C.softmax_cross_entropy(prediction, label)
        test_case.assertTrue(
            "The dimension of prediction must be greater than or equal to 2, but found"
            in str(context.exception)
        )

    def test_softmax_cross_entropy_prediction_shape_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            prediction = flow.randn(1, 10)
            label = flow.randn(1, 11)
            flow._C.softmax_cross_entropy(prediction, label)
        test_case.assertTrue(
            "must match the size of prediction" in str(context.exception)
        )

    def test_softmax_cross_entropy_dtype_err(test_case):
        with test_case.assertRaises(TypeError) as context:
            prediction = flow.randn(1, 10, dtype=flow.float32)
            label = flow.randn(1, 10, dtype=flow.float64)
            flow._C.softmax_cross_entropy(prediction, label)
        test_case.assertTrue(
            "label and prediction are expected to have the same dtype, but found"
            in str(context.exception)
        )

    def test_softmax_cross_entropy_grad_prob_numaxes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            dy = flow.randn(10, 5)
            label = flow.randn(10, 10, 5)
            prob = flow.randn(10)
            flow._C.softmax_cross_entropy_grad(dy, label, prob)
        test_case.assertTrue(
            "The dimension of prob must be greater than or equal to 2, but found "
            in str(context.exception)
        )

    def test_softmax_cross_entropy_grad_dy_numaxes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            dy = flow.randn(10, 10, 5)
            label = flow.randn(10, 10, 5)
            prob = flow.randn(10, 10, 5)
            flow._C.softmax_cross_entropy_grad(dy, label, prob)
        test_case.assertTrue(
            "The dimension of dy is expected to be less than that of prob by 1, but found"
            in str(context.exception)
        )

    def test_softmax_cross_entropy_grad_dy_i_shape_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            dy = flow.randn(10, 8)
            label = flow.randn(10, 10, 5)
            prob = flow.randn(10, 10, 5)
            flow._C.softmax_cross_entropy_grad(dy, label, prob)
        test_case.assertTrue("must match the size of label" in str(context.exception))

    def test_softmax_cross_entropy_grad_prob_shape_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            dy = flow.randn(10, 10)
            label = flow.randn(10, 10, 5)
            prob = flow.randn(10, 10, 6)
            flow._C.softmax_cross_entropy_grad(dy, label, prob)
        test_case.assertTrue("must match the size of prob" in str(context.exception))

    def test_softmax_cross_entropy_grad_label_dtype_err(test_case):
        with test_case.assertRaises(TypeError) as context:
            dy = flow.randn(10, 10, dtype=flow.float64)
            label = flow.randn(10, 10, 5, dtype=flow.float32)
            prob = flow.randn(10, 10, 5, dtype=flow.float64)
            flow._C.softmax_cross_entropy_grad(dy, label, prob)
        test_case.assertTrue(
            "label and prob are expected to have the same dtype, but found"
            in str(context.exception)
        )

    def test_softmax_cross_entropy_grad_dy_dtype_err(test_case):
        with test_case.assertRaises(TypeError) as context:
            dy = flow.randn(10, 10, dtype=flow.float32)
            label = flow.randn(10, 10, 5, dtype=flow.float64)
            prob = flow.randn(10, 10, 5, dtype=flow.float64)
            flow._C.softmax_cross_entropy_grad(dy, label, prob)
            print(str(context.exception))
        test_case.assertTrue(
            "dy and prob are expected to have the same dtype, but found"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
