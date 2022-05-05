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

from oneflow.test_utils.automated_test_util import *


class TestNormalizationError(flow.unittest.TestCase):
    def test_normalization_moving_mean_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 4, 2, 2), dtype=flow.float32)
            moving_mean = flow.ones((4,), dtype=flow.float32)
            weight = flow.ones((4,), dtype=flow.float32)
            bias = flow.ones((4,), dtype=flow.float32)

            out = flow._C.normalization(
                x, moving_mean, None, weight, bias, 1, 1e-5, 0.9, False
            )

        test_case.assertTrue(
            "Check failed: (moving_mean && moving_variance) || (!moving_mean && !moving_variance) Both moving_mean and moving_variance should be None or Tensor."
            in str(ctx.exception)
        )

    def test_normalization_x_input_axes_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1,), dtype=flow.float32)
            weight = flow.ones((4,), dtype=flow.float32)
            bias = flow.ones((4,), dtype=flow.float32)

            out = flow._C.normalization(
                x, None, None, weight, bias, 1, 1e-5, 0.9, False
            )

        test_case.assertTrue(
            "Check failed: (x->shape()->NumAxes()) >= (2) (1 vs 2) NumAxes of x should be greater or equal than 2."
            in str(ctx.exception)
        )

    def test_normalization_eval_need_moving_statistic_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 2,), dtype=flow.float32)
            weight = flow.ones((2,), dtype=flow.float32)
            bias = flow.ones((2,), dtype=flow.float32)

            out = flow._C.normalization(
                x, None, None, weight, bias, 1, 1e-5, 0.9, False
            )

        test_case.assertTrue(
            "Check failed: moving_mean && moving_variance Must have moving_mean and moving_variance in eval mode."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
