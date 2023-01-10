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


class TestSmoothL1LossError(flow.unittest.TestCase):
    def test_smooth_l1_loss_shape_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            input = flow.randn(10)
            target = flow.randn(11)
            reduction = "mean"
            beta = 1.0
            flow._C.smooth_l1_loss(input, target, beta, reduction)
        test_case.assertTrue("must match the size of target" in str(context.exception))

    def test_smooth_l1_loss_beta_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            input = flow.randn(10)
            target = flow.randn(10)
            reduction = "mean"
            beta = -1.0
            flow._C.smooth_l1_loss(input, target, beta, reduction)
        test_case.assertTrue(
            "beta must be greater than or equal to 0" in str(context.exception)
        )

    def test_smooth_l1_loss_dtype_err(test_case):
        with test_case.assertRaises(TypeError) as context:
            input = flow.randn(10, dtype=flow.float32)
            target = flow.randn(10, dtype=flow.float64)
            reduction = "mean"
            beta = 1.0
            flow._C.smooth_l1_loss(input, target, beta, reduction)
        test_case.assertTrue(
            "input and target are expected to have the same dtype"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
