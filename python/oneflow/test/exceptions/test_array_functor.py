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
import oneflow.unittest
import oneflow as flow


class TestArrayError(flow.unittest.TestCase):
    def test_argmax_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            y = flow.argmax(x, dim=4)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_broadcast_like_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 0), dtype=flow.float32, requires_grad=True)
            like = flow.ones((2, 2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.broadcast_like(x, like)
        test_case.assertTrue(
            "The expanded size of the tensor" in str(context.exception)
        )

    def test_broadcast_like_numaxes_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((2, 2, 2), dtype=flow.float32, requires_grad=True)
            like = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            y = flow._C.broadcast_like(x, like)
        print(str(context.exception))
        test_case.assertTrue("The number of sizes provided" in str(context.exception))

    def test_concat_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.concat([x1, x2], dim=3)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_concat_dim_equal_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2, 2), dtype=flow.float32, requires_grad=True)
            y = flow.concat([x1, x2])
        test_case.assertTrue(
            "Tensors must have same number of dimensions" in str(context.exception)
        )

    def test_concat_match_size_runtime_error(test_case):
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
            "stack expects each tensor to be equal size" in str(context.exception)
        )

    def test_expand_dim_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 1), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2), dtype=flow.float32, requires_grad=True)
            y = flow.expand(x1, x2.shape)
        test_case.assertTrue(
            "be greater or equal to the number of dimensions in the tensor"
            in str(context.exception)
        )

    def test_expand_g_shape_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 4), dtype=flow.float32, requires_grad=True)
            y = flow.expand(x1, x2.shape)
        test_case.assertTrue(
            "The expanded size of the tensor" in str(context.exception)
        )

    def test_expand_l_shape_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 0), dtype=flow.float32, requires_grad=True)
            y = flow.expand(x1, x2.shape)
        test_case.assertTrue(
            "The expanded size of the tensor" in str(context.exception)
        )

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
            "shifts and dimensions must align" in str(context.exception)
        )

    def test_gather_index_type_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2), dtype=flow.float32)
            y = flow.gather(x1, 1, x2)
        test_case.assertTrue(
            "gather(): Expected dtype int32 or int64 for index"
            in str(context.exception)
        )

    def test_gather_dim_value_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2), dtype=flow.int64)
            y = flow.gather(x1, 2, x2)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_gather_dim_equal_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2, 2, 2), dtype=flow.int64)
            y = flow.gather(x1, 1, x2)
        test_case.assertTrue(
            "Index tensor must have the same number of dimensions as input tensor"
            in str(context.exception)
        )

    def test_gather_size_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2, 2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((4, 2), dtype=flow.int64)
            y = flow.gather(x1, 1, x2)
        test_case.assertTrue(
            "Size does not match at dimension" in str(context.exception)
        )

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

    def test_narrow_dim_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((3, 3), dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 3, 0, 2)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_narrow_0_dim_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(1, dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 0, 0, 0)
        test_case.assertTrue(
            "narrow() cannot be applied to a 0-dim tensor." in str(context.exception)
        )

    def test_narrow_start_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((3, 3), dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 0, 4, 0)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_narrow_length_exceed_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((3, 3), dtype=flow.float32, requires_grad=True)
            y = flow.narrow(x, 0, 2, 2)
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
            "diagonal dimensions cannot be identical" in str(context.exception)
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

    def test_chunk_tensor_dim_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(1, dtype=flow.float32, requires_grad=True)
            y = flow.chunk(x, chunks=2, dim=4)
        test_case.assertTrue(
            "chunk expects at least a 1-dimensional tensor" in str(context.exception)
        )

    def test_chunk_value_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.chunk(x, chunks=-1, dim=4)
        test_case.assertTrue(
            "chunk expects `chunks` to be greater than 0, got" in str(context.exception)
        )

    def test_meshgrid_tensors_scalar_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.tensor([], dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((1, 2, 3), dtype=flow.float32, requires_grad=True)
            y = flow.meshgrid(x1, x2)
        test_case.assertTrue(
            "Expected scalar or 1D tensor in the tensor list" in str(context.exception)
        )

    def test_meshgrid_tensors_size_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            y = flow.meshgrid([])
        test_case.assertTrue(
            "meshgrid expects a non-empty TensorList" in str(context.exception)
        )

    def test_meshgrid_tensors_dtype_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2), dtype=flow.float16, requires_grad=True)
            y = flow.meshgrid(x1, x2)
        test_case.assertTrue(
            "meshgrid expects all tensors to have the same dtype"
            in str(context.exception)
        )

    def test_meshgrid_tensors_placement_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.tensor(
                [0.0, 1.0],
                dtype=flow.float32,
                placement=flow.placement("cpu", ranks=[0]),
                sbp=[flow.sbp.broadcast],
            )
            x2 = flow.tensor(
                [0.0, 1.0],
                dtype=flow.float32,
                placement=flow.placement("cpu", ranks=[0]),
                sbp=[flow.sbp.broadcast],
            ).to_local()
            y = flow.meshgrid(x1, x2)
        test_case.assertTrue(
            "meshgrid expects all tensors are global tensor" in str(context.exception)
        )

    def test_meshgrid_indexing_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x1 = flow.ones((2), dtype=flow.float32, requires_grad=True)
            x2 = flow.ones((2), dtype=flow.float32, requires_grad=True)
            y = flow.meshgrid(x1, x2, indexing="ab")
        test_case.assertTrue(
            "meshgrid: indexing must be one of" in str(context.exception)
        )

    def test_index_select_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=flow.float32, requires_grad=True
            )
            index = flow.tensor([0, 1], dtype=flow.float32)
            y = flow.index_select(x, 1, index)
        test_case.assertTrue(
            "Expected dtype int32 or int64 for index" in str(context.exception)
        )

    def test_index_select_index_num_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=flow.float32, requires_grad=True
            )
            index = flow.tensor([[0]], dtype=flow.int32)
            y = flow.index_select(x, 1, index)
        test_case.assertTrue(
            "Index is supposed to be a vector" in str(context.exception)
        )

    def test_index_select_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=flow.float32, requires_grad=True
            )
            index = flow.tensor([0], dtype=flow.int32)
            y = flow.index_select(x, 4, index)
        test_case.assertTrue("Dimension out of range" in str(context.exception))

    def test_to_device_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor(
                [0.0, 1.0],
                dtype=flow.float32,
                placement=flow.placement("cpu", ranks=[0]),
                sbp=[flow.sbp.split(0)],
            )
            x.to("cpp")
        test_case.assertTrue(
            "Only string device without device id" in str(context.exception)
        )

    def test_to_other_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([0.0, 1.0], dtype=flow.float32)
            other = flow.tensor(
                [0.0, 1.0],
                dtype=flow.float32,
                placement=flow.placement("cpu", ranks=[0]),
                sbp=[flow.sbp.split(0)],
            )
            x.to(other)
        test_case.assertTrue(
            "tensor.to(other) can only be called when tensor and other are local tensors"
            in str(context.exception)
        )

    def test_in_top_k_num_equal_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            target = flow.tensor([[3, 1]], dtype=flow.int32)
            prediction = flow.tensor(
                [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=flow.float32
            )
            out = flow.in_top_k(target, prediction, k=1)
        test_case.assertTrue(
            "The num of targets must equal the num of predictions"
            in str(context.exception)
        )

    def test_in_top_k_targets_dim_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            target = flow.tensor([[3, 1], [1, 3]], dtype=flow.int32)
            prediction = flow.tensor(
                [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=flow.float32
            )
            out = flow.in_top_k(target, prediction, k=1)
        test_case.assertTrue(
            "The dimension of targets must be 1" in str(context.exception)
        )

    def test_in_top_k_pre_dim_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            target = flow.tensor([3, 1], dtype=flow.int32)
            prediction = flow.tensor(
                [[[0.0, 1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0, 0.0]]], dtype=flow.float32
            )
            out = flow.in_top_k(target, prediction, k=1)
        test_case.assertTrue(
            "The dimension of predictions must be 2" in str(context.exception)
        )

    def test_repeat_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1], [1]], dtype=flow.int32)
            y = x.repeat(1)
        test_case.assertTrue(
            "Number of dimensions of repeat dims can not be" in str(context.exception)
        )

    def test_tile_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[1], [1]], dtype=flow.int32)
            y = x.tile(-1)
        test_case.assertTrue(
            "Trying to create tensor with negative dimension" in str(context.exception)
        )

    def test_t_runtime_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.tensor([[[1]]], dtype=flow.int32)
            y = x.t()
        test_case.assertTrue(
            "t() expects a tensor with <= 2 dimensions" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
