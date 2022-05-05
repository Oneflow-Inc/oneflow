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


class TestFusedMLPError(flow.unittest.TestCase):
    def test_fuse_mlp_weight_size_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            bias = flow.ones((4,), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [], [bias], False)

        test_case.assertTrue(
            "Check failed: (weight_size) >= (1) (0 vs 1) The number of weights should be greater equal than 1"
            in str(ctx.exception)
        )

    def test_fuse_mlp_weight_bias_size_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((4, 4), dtype=flow.float32)
            w2 = flow.ones((4, 4), dtype=flow.float32)
            bias1 = flow.ones((4,), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1, w2], [bias1], False)

        test_case.assertTrue(
            "Check failed: (weight_size) == (bias_size) (2 vs 1) The number of weights should be equal to biases"
            in str(ctx.exception)
        )

    def test_fuse_mlp_weight_numaxes_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((4,), dtype=flow.float32)
            bias1 = flow.ones((4,), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)
        test_case.assertTrue(
            "Check failed: (weight_shape->NumAxes()) == (2) (1 vs 2) Weight's dim should == 2"
            in str(ctx.exception)
        )

    def test_fuse_mlp_bias_numaxes_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((4, 4), dtype=flow.float32)
            bias1 = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)
        test_case.assertTrue(
            "Check failed: (bias_shape->NumAxes()) == (1) (2 vs 1) Bias's dim should == 1"
            in str(ctx.exception)
        )

    def test_fuse_mlp_bias_first_dim_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((6, 4), dtype=flow.float32)
            bias1 = flow.ones((5), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)

        test_case.assertTrue(
            "Check failed: (bias_shape->At(0)) == (n) (5 vs 6) Bias's dim is not equal to weight's first dim."
            in str(ctx.exception)
        )

    def test_fuse_mlp_weight_second_dim_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((2, 4), dtype=flow.float32)
            w1 = flow.ones((3, 6), dtype=flow.float32)
            bias1 = flow.ones((3), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)

        test_case.assertTrue(
            "Check failed: (weight_shape->At(1)) == (k) (6 vs 4) weight's second dim should be equal to input's second dim."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
