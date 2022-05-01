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
from tkinter import X
import unittest
from collections import OrderedDict

import os
import numpy as np
import time
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestArrayError(flow.unittest.TestCase):
    def test_argmax_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            y = flow.argmax(x, dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_broadcast_like_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            like = flow.ones((2, 4), dtype=flow.float32, requires_grad=True)
            broadcast_axes = [4]
            y = flow.broadcast_like(x, like, broadcast_axes)
        test_case.assertTrue("broadcast_axes out of range" in str(context.exception))

    def test_broadcast_like_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            like = flow.ones((2, 4), dtype=flow.float32, requires_grad=True)
            y = flow.broadcast_like(x, like)
        test_case.assertTrue(
            "doesn't match the broadcast shape" in str(context.exception)
        )

    def test_concat_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.concat([x1, x2], dim=3)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_concat_runtime_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.concat([x1, x2])
        test_case.assertTrue(
            "Tensors must have same number of dimensions" in str(context.exception)
        )

    def test_concat_runtime_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.concat([x1, x2])
        test_case.assertTrue(
            "Sizes of tensors must match except in dimension" in str(context.exception)
        )

    def test_stack_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            y = flow.concat([x1, x2], dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_stack_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.stack([x1, x2])
        test_case.assertTrue(
            "Stacks expects each tensor to be equal size" in str(context.exception)
        )

    def test_expand_runtime_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2), dtype=flow.float32, requires_grad=True)
            y = flow.expand(x1, x2.shape)
        test_case.assertTrue(
            "The desired expanded dims should not be less than the input dims"
            in str(context.exception)
        )

    def test_expand_runtime_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 4), dtype=flow.float32, requires_grad=True)
            y = flow.expand(x1, x2.shape)
        test_case.assertTrue("Invalid expand shape" in str(context.exception))

    def test_expand_runtime_error3(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 0), dtype=flow.float32, requires_grad=True)
            y = flow.expand(x1, x2.shape)
        test_case.assertTrue("Invalid expand shape" in str(context.exception))

    def test_squeeze_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            y = flow.squeeze(x, dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_roll_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.roll(x, [0, 1], [0])
        test_case.assertTrue(
            "parameters should have the same size" in str(context.exception)
        )

    def test_gather_runtime_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.gather(x1, 2, x2)
        test_case.assertTrue("Value of dim is out of range" in str(context.exception))

    def test_gather_runtime_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.gather(x1, 1, x2)
        test_case.assertTrue(
            "Dimensions of input and index should equal" in str(context.exception)
        )

    def test_gather_runtime_error3(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((4, 2), dtype=flow.float32, requires_grad=True)
            y = flow.gather(x1, 1, x2)
        test_case.assertTrue("index.size(d) <= input.size(d)" in str(context.exception))

    def test_tensor_scatter_nd_update_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.arange(8, dtype=flow.float32, requires_grad=True)
            indices = flow.tensor([[1], [3], [5]])
            updates = flow.tensor([-1, -2, -3], dtype=flow.float64, requires_grad=True)
            y = flow.tensor_scatter_nd_update(x, indices, updates)
        test_case.assertTrue(
            "The dtype of tensor and updates must be same." in str(context.exception)
        )

    def test_view_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 3, 4), dtype=flow.float32, requires_grad=True).permute(
                1, 0, 2
            )
            x2 = flow.ones((4, 6), dtype=flow.float32, requires_grad=True)
            y = flow.view(x1, x2.shape)
        test_case.assertTrue(
            "view size is not compatible with input tensor's size"
            in str(context.exception)
        )

    def test_narrow_index_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((3, 3), dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 3, 0, 2)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_narrow_index_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((3, 3), dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 0, 3, 2)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_narrow_runtime_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(1, dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 0, 0, 0)
        test_case.assertTrue(
            "narrow() cannot be applied to a 0-dim tensor." in str(context.exception)
        )

    def test_narrow_runtime_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 1), dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 0, 0, 2)
        test_case.assertTrue("exceeds dimension size" in str(context.exception))

    def test_diagonal_index_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.diagonal(x, 1, 3, 2)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_diagonal_index_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.diagonal(x, 1, 2, 3)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_diagonal_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.diagonal(x, 1, 2, 2)
        test_case.assertTrue(
            "Diagonal dimensions cannot be identical" in str(context.exception)
        )

    def test_split_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.split(x, split_size_or_sections=0, dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_split_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.split(x, split_size_or_sections=-1)
        test_case.assertTrue(
            "split expects split_size be non-negative, but got split_size"
            in str(context.exception)
        )

    def test_splitwithsize_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((5, 2), dtype=flow.float32, requires_grad=True)
            y = flow.split(x, [1, 3])
        test_case.assertTrue(
            "split_with_sizes expects split_sizes to sum exactly to "
            in str(context.exception)
        )

    def test_unbind_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.unbind(x, dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_chunk_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.chunk(x, chunks=2, dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_chunk_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(1, dtype=flow.float32, requires_grad=True)
            y = flow.chunk(x, chunks=2, dim=4)
        test_case.assertTrue(
            "chunk expects at least a 1-dimensional tensor" in str(context.exception)
        )

    def test_chunk_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.chunk(x, chunks=-1, dim=4)
        test_case.assertTrue(
            "chunk expects `chunks` to be greater than 0, got" in str(context.exception)
        )

    def test_meshgrid_runtime_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.tensor([], dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.meshgrid(x1, x2)
        test_case.assertTrue(
            "Expected scalar or 1D tensor in the tensor list" in str(context.exception)
        )

    def test_meshgrid_runtime_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.tensor(1, dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.meshgrid(x1)
        print(context.exception)
        test_case.assertTrue(
            "Meshgrid expects a non-empty TensorList" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
