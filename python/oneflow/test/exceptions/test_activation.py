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
from collections import OrderedDict

import os
import numpy as np
import time
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestActivationError(flow.unittest.TestCase):
    def test_relu_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            x.relu_()
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_prelu_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            m = flow.nn.PReLU(5)
            y = m(x)
        test_case.assertTrue(
            "num_parameters in prelu must be 1 or 4" in str(context.exception)
        )

    def test_celu_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            m = flow.nn.CELU(alpha=1.0, inplace=True)
            y = m(x)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_glu_scalar_tensor_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(1.0)
            m = flow.nn.GLU()
            y = m(x)
        test_case.assertTrue(
            "glu does not support scalars because halving size must be even"
            in str(context.exception)
        )

    def test_glu_dim_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2, 4)
            m = flow.nn.GLU(dim=3)
            y = m(x)
        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-2, 1], but got 3)"
            in str(context.exception)
        )

    def test_glu_dim_even_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2, 3)
            m = flow.nn.GLU()
            y = m(x)
        test_case.assertTrue(
            "Halving dimension must be even, but dimension 1 is size 3"
            in str(context.exception)
        )

    def test_hard_sigmoid_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2)
            x.requires_grad = True
            m = flow.nn.Hardsigmoid(inplace=True)
            y = m(x)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_hard_shrink_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2)
            x.requires_grad = True
            m = flow.nn.Hardshrink(inplace=True)
            y = m(x)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_softmax_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2, 4)
            m = flow.nn.Softmax(dim=2)
            y = m(x)
        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-2, 1], but got 2)"
            in str(context.exception)
        )

    def test_soft_shrink_inplace_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2)
            x.requires_grad = True
            m = flow.nn.Softshrink(inplace=True)
            y = m(x)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_soft_shrink_alpha_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.randn(2)
            x.requires_grad = True
            m = flow.nn.Softshrink(-0.1)
            y = m(x)
        test_case.assertTrue(
            "alpha must be greater or equal to 0, but found to be -0.1."
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
