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


class TestDeformConv(flow.unittest.TestCase):
    def test_deform_conv2d_invalid_input_sizes(test_case):
        input = flow.randn(2, 5, 1)
        weight = flow.randn(2, 5, 1, 1)
        offset = flow.randn(2, 5, 1, 1)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight)
        test_case.assertTrue(
            "The dimension of input tensor weight must be " in str(ctx.exception)
        )

    def test_deform_conv2d_invalid_offset_sizes(test_case):
        input = flow.randn(2, 5, 1, 1)
        weight = flow.randn(2, 5, 1, 1)
        offset = flow.randn(2, 5, 1)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight)
        test_case.assertTrue(
            "The dimension of offset tensor weight must be " in str(ctx.exception)
        )

    def test_deform_conv2d_invalid_weight_sizes(test_case):
        input = flow.randn(2, 5, 1, 1)
        weight = flow.randn(2, 5, 5)
        offset = flow.randn(2, 3, 1, 1)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight)
        test_case.assertTrue(
            "The dimension of weight tensor weight must be " in str(ctx.exception)
        )

    def test_deform_conv2d_invalid_mask_sizes(test_case):
        input = flow.randn(2, 5, 1, 1)
        weight = flow.randn(2, 4, 1, 1)
        offset = flow.randn(2, 3, 1, 1)
        mask = flow.randn(2, 3, 1)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight, mask=mask)
        test_case.assertTrue(
            "The dimension of mask tensor weight must be" in str(ctx.exception)
        )

    def test_deform_conv2d_invalid_dilation_parm(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 18, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(
                input, offset, weight, dilation=(-1, 0)
            )
        test_case.assertTrue("The dilation must be greater than" in str(ctx.exception))

    def test_deform_conv2d_invalid_pad_parm(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 18, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(
                input, offset, weight, padding=(-1, 0)
            )
        test_case.assertTrue("The pad must be greater than" in str(ctx.exception))

    def test_deform_conv2d_invalid_stride_parm(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 18, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(
                input, offset, weight, stride=(-1, 0)
            )
        test_case.assertTrue("The stride must be greater than" in str(ctx.exception))

    def test_deform_conv2d_invalid_offset_shape(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 9, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight)
        test_case.assertTrue(
            "The shape of the offset tensor at dimension 1 is not valid"
            in str(ctx.exception)
        )

    def test_deform_conv2d_invalid_batch_size(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(3, 18, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight)
        test_case.assertTrue("invalid batch size of offset" in str(ctx.exception))

    def test_deform_conv2d_invalid_mask_shape(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 18, 8, 8)
        mask = flow.randn(4, 1, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(input, offset, weight, mask=mask)
        test_case.assertTrue("mask.shape[1] is not valid" in str(ctx.exception))

    def test_deform_conv2d_invalid_output_size(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 18, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(
                input, offset, weight, dilation=(10, 10)
            )
        test_case.assertTrue("Calculated output size too small" in str(ctx.exception))

    def test_deform_conv2d_invalid_offset_output_dims(test_case):
        input = flow.randn(4, 3, 10, 10)
        weight = flow.randn(5, 3, 3, 3)
        offset = flow.randn(4, 18, 8, 8)
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.nn.functional.deform_conv2d(
                input, offset, weight, dilation=(2, 2)
            )
        test_case.assertTrue("invalid offset output dims" in str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
