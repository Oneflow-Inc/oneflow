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
import re
import unittest

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestBiasAddError(flow.unittest.TestCase):
    def test_bias_add_dimension_match_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            bias = flow.ones((5,), dtype=flow.float32)
            out = flow._C.bias_add(x, bias, axis=1)

        test_case.assertTrue(
            "The size of tensor x (4,4) must match the size of tensor b (5,) at dimension 1"
            in str(ctx.exception)
        )

    def test_bias_add_index_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            bias = flow.ones((5,), dtype=flow.float32)
            out = flow._C.bias_add(x, bias, axis=3)

        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [-2,1], but got 3)"
            in str(ctx.exception)
        )


class TestCrossEntropyError(flow.unittest.TestCase):
    def test_cross_entropy_reduction_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            target = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.cross_entropy(x, target, None, 0, "just_test")

        test_case.assertTrue(
            "Reduction should be none, sum or mean." in str(ctx.exception)
        )


class TestCTCLossError(flow.unittest.TestCase):
    def test_ctcloss_reduction_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((5, 2, 3), dtype=flow.float32)
            targets = flow.tensor([[1, 2, 2], [1, 2, 2]], dtype=flow.int32)
            input_lengths = flow.tensor([5, 5], dtype=flow.int32)
            target_lengths = flow.tensor([3, 3], dtype=flow.int32)
            max_target_length = 0
            if targets.ndim == 1:
                max_target_length = target_lengths.max().item()
            elif targets.ndim == 2:
                max_target_length = targets.shape[1]
            loss = flow._C.ctc_loss(
                x,
                targets,
                input_lengths,
                target_lengths,
                max_target_length,
                blank=0,
                zero_infinity=False,
                reduction="just_test",
            )
        test_case.assertTrue(
            "Reduction should be none, sum or mean." in str(ctx.exception)
        )


class TestPadError(flow.unittest.TestCase):
    def test_pad_size_attribute_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1), dtype=flow.float32)
            out = flow._C.pad(x, (1, 1, 1, 1, 1))
        test_case.assertTrue(
            "Pad size should less than or equal to input axes * 2."
            in str(ctx.exception)
        )

    def test_pad_size_mod2_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1), dtype=flow.float32)
            out = flow._C.pad(x, (1, 1, 1,))

        test_case.assertTrue(
            "Length of pad must be even but instead it equals 3" in str(ctx.exception)
        )

    def test_reflect_pad_size_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1, 2, 2), dtype=flow.float32)
            out = flow._C.pad(x, (4, 4, 4, 4), mode="reflect")

        test_case.assertTrue(
            "Padding size should be less than the corresponding input dimension, but got:"
            in str(ctx.exception)
        )

    def test_pad_mode_error(test_case):
        with test_case.assertRaises(NotImplementedError) as ctx:
            x = flow.ones((1, 1, 2, 2), dtype=flow.float32)
            out = flow._C.pad(x, (4, 4, 4, 4), mode="test")

        test_case.assertTrue(
            "Pad mode is test, but only constant, reflect and replicate are valid."
            in str(ctx.exception)
        )


class TestFusedMLPError(flow.unittest.TestCase):
    def test_fuse_mlp_weight_size_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            bias = flow.ones((4,), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [], [bias], False)

        test_case.assertTrue(
            "The number of weights should be greater equal than 1" in str(ctx.exception)
        )

    def test_fuse_mlp_weight_bias_size_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((4, 4), dtype=flow.float32)
            w2 = flow.ones((4, 4), dtype=flow.float32)
            bias1 = flow.ones((4,), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1, w2], [bias1], False)

        test_case.assertTrue(
            "The number of weights should be equal to biases" in str(ctx.exception)
        )

    def test_fuse_mlp_weight_numaxes_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((4,), dtype=flow.float32)
            bias1 = flow.ones((4,), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)
        test_case.assertTrue("Weight's dim size should == 2" in str(ctx.exception))

    def test_fuse_mlp_bias_numaxes_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((4, 4), dtype=flow.float32)
            bias1 = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)
        test_case.assertTrue("Bias's dim size should == 1" in str(ctx.exception))

    def test_fuse_mlp_bias_first_dim_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w1 = flow.ones((6, 4), dtype=flow.float32)
            bias1 = flow.ones((5), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)

        test_case.assertTrue(
            "Bias's dim is not equal to weight's first dim." in str(ctx.exception)
        )

    def test_fuse_mlp_weight_second_dim_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((2, 4), dtype=flow.float32)
            w1 = flow.ones((3, 6), dtype=flow.float32)
            bias1 = flow.ones((3), dtype=flow.float32)
            out = flow._C.fused_mlp(x, [w1,], [bias1,], False)

        test_case.assertTrue(
            "weight's second dim should be equal to input's second dim."
            in str(ctx.exception)
        )


class TestL2NormalizeError(flow.unittest.TestCase):
    def test_l2normalize_axis_error1(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((3, 3), dtype=flow.float32)
            out = flow._C.normalize(x, dim=3, use_l2_norm_kernel=True)
        test_case.assertTrue("Axis should < 2 but axis is 3 now." in str(ctx.exception))

    def test_l2normalize_axis_error2(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((3, 3), dtype=flow.float32)
            out = flow._C.normalize(x, dim=-3, use_l2_norm_kernel=True)
        test_case.assertTrue(
            "Axis should >=0 but axis is -1 now." in str(ctx.exception)
        )


class TestLossBaseFunctorError(flow.unittest.TestCase):
    def test_loss_base_reduction_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            target = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.mse_loss(x, target, "just_test")

        test_case.assertTrue(
            "Reduction should be none, sum or mean." in str(ctx.exception)
        )


class TestMatmulError(flow.unittest.TestCase):
    def test_matmul_dimension_error1(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((), dtype=flow.float32)
            w = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.matmul(x, w, False, False, 1.0)
        test_case.assertTrue("Tensor a's dim should >= 1" in str(ctx.exception))

    def test_matmul_dimension_error2(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w = flow.ones((), dtype=flow.float32)
            out = flow._C.matmul(x, w, False, False, 1.0)
        test_case.assertTrue("Tensor b's dim should >= 1" in str(ctx.exception))


class TestPixelShuffleError(flow.unittest.TestCase):
    def test_pixel_shuffle_4D_input_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 8, 4, 4, 1), dtype=flow.float32)
            out = flow._C.pixel_shuffle(x, 2, 2)

        test_case.assertTrue("Only Accept 4D Tensor" in str(ctx.exception))

    def test_pixel_shuffle_channel_divisble_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 8, 4, 4), dtype=flow.float32)
            out = flow._C.pixel_shuffle(x, 2, 3)

        test_case.assertTrue(
            "The channels of input tensor must be divisible by (upscale_factor * upscale_factor) or (h_upscale_factor * w_upscale_factor)"
            in str(ctx.exception)
        )


class TestTripletMarginLossError(flow.unittest.TestCase):
    def test_triplet_margin_loss_reduce_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            anchor = flow.ones((3, 3), dtype=flow.float32)
            positive = flow.ones((3, 3), dtype=flow.float32)
            negative = flow.ones((3, 3), dtype=flow.float32)

            triplet_loss = flow._C.triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=0.001,
                p=2,
                eps=1e-5,
                swap=False,
                reduction="just_test",
            )

        test_case.assertTrue(
            "Reduction should be none, sum or mean." in str(ctx.exception)
        )


class TestNormalError(flow.unittest.TestCase):
    def test_normal_data_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow._C.normal(mean=0.0, std=1.0, size=(3, 3), dtype=flow.int32)

        test_case.assertTrue(
            "Only support float and double in normal()." in str(ctx.exception)
        )

    def test_normal_out_tensor_data_type_error(test_case):
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.zeros((3, 3), dtype=flow.float64)
            x = flow._C.normal(
                mean=0.0, std=1.0, size=(3, 3), dtype=flow.float32, out=out
            )

        test_case.assertTrue(
            "data type oneflow.float32 does not match data type of out parameter oneflow.float64"
            in str(ctx.exception)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_normal_out_tensor_device_type_error(test_case):
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.zeros((3, 3), dtype=flow.float32, device="cuda")
            x = flow._C.normal(
                mean=0.0,
                std=1.0,
                size=(3, 3),
                dtype=flow.float32,
                out=out,
                device="cpu",
            )

        test_case.assertTrue(
            "does not match device type of out parameter" in str(ctx.exception)
        )


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
            "Both moving_mean and moving_variance should be None or Tensor."
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
            "NumAxes of x should be greater or equal than 2." in str(ctx.exception)
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
            "Must have moving_mean and moving_variance in eval mode."
            in str(ctx.exception)
        )


class TestOnehotError(flow.unittest.TestCase):
    def test_onehot_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((3, 3), dtype=flow.float32)
            out = flow._C.one_hot(x, 3, 0.9, 0)

        test_case.assertTrue(
            "one_hot is only applicable to index tensor." in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
