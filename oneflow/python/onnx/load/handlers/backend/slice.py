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
import numpy as np
import tensorflow as tf

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("Slice")
@tf_func(tf.strided_slice)
class Slice(BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]

        full_sizes = x.get_shape().as_list()
        full_begin = [0] * len(full_sizes)

        starts = node.attrs.get("starts")
        ends = node.attrs.get("ends")
        slice_len = len(starts)
        axes = node.attrs.get("axes", list(range(slice_len)))

        for i in range(slice_len):
            starts[i] = full_sizes[axes[i]] + starts[i] if starts[i] < 0 else starts[i]
            ends[i] = full_sizes[axes[i]] + ends[i] if ends[i] < 0 else ends[i]
            if full_sizes[axes[i]] is not None:
                ends[i] = np.min([full_sizes[axes[i]], ends[i]])
                starts[i] = np.min([full_sizes[axes[i]], starts[i]])
            full_begin[axes[i]] = starts[i]
            full_sizes[axes[i]] = ends[i] - starts[i]

        return [
            cls.make_tensor_from_onnx_node(
                node,
                tf_func=tf.slice,
                inputs=[
                    tensor_dict[node.inputs[0]],
                    tf.constant(full_begin),
                    tf.constant(full_sizes),
                ],
                **kwargs
            )
        ]

    @classmethod
    def version_10(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        input_tensor = tensor_dict[node.inputs[0]]
        starts = tensor_dict[node.inputs[1]]
        ends = tensor_dict[node.inputs[2]]

        # first of all, get the input tensor shape
        input_tensor_shape = tf.shape(input_tensor, out_type=ends.dtype)

        axes = (
            tensor_dict[node.inputs[3]]
            if len(node.inputs) >= 4
            else tf.range(tf.shape(starts)[0], dtype=ends.dtype)
        )

        is_axes_negative = tf.less(axes, tf.zeros_like(axes))
        axes = tf.where(
            is_axes_negative, axes + tf.cast(tf.rank(input_tensor), axes.dtype), axes
        )

        # expand a dimension of 1 at the end
        sparse_indices = tf.expand_dims(axes, -1)

        # build the indexed dimension sizes as sparse_shape
        sparse_shape = tf.gather_nd(params=input_tensor_shape, indices=sparse_indices)
        sparse_shape = tf.cast(sparse_shape, ends.dtype)

        # take care of starts, ends that are larger than the dim size.
        starts_min = tf.minimum(starts, sparse_shape)
        ends_min = tf.minimum(ends, sparse_shape)

        # take care of starts, ends that are negative
        is_starts_negative = tf.less(starts_min, tf.zeros_like(starts_min))
        starts_final = tf.where(
            is_starts_negative, starts_min + sparse_shape, starts_min
        )
        is_ends_negative = tf.less(ends_min, tf.zeros_like(ends_min))
        ends_final = tf.where(is_ends_negative, ends_min + sparse_shape, ends_min)

        # need to densify everything for the inputs to slice
        # the output shape is the input_tensor rank
        output_shape = tf.reshape(tf.rank(input_tensor), [1])
        output_shape = tf.cast(output_shape, ends.dtype)

        # create dense tensor, pad 0 as default begins
        dense_begins = tf.compat.v1.sparse_to_dense(
            sparse_indices, output_shape, starts_final
        )
        # create dense tensor, pad -1 for next step
        dense_ends = tf.compat.v1.sparse_to_dense(
            sparse_indices,
            output_shape,
            ends_final,
            default_value=tf.constant(-1, dtype=dense_begins.dtype),
        )
        # replace -1 with respective dimension sizes
        dense_ends = tf.where(
            tf.equal(dense_ends, tf.constant(-1, dtype=dense_begins.dtype)),
            input_tensor_shape,
            dense_ends,
        )

        # create dense tensor for steps if not already so
        if len(node.inputs) >= 5:
            dense_steps = tf.compat.v1.sparse_to_dense(
                sparse_indices,
                output_shape,
                tensor_dict[node.inputs[4]],
                default_value=tf.constant(1, dtype=tensor_dict[node.inputs[4]].dtype),
            )
        else:
            dense_steps = tf.ones(input_tensor_shape.shape, ends.dtype)

        return [
            cls.make_tensor_from_onnx_node(
                node,
                inputs=[
                    tensor_dict[node.inputs[0]],
                    dense_begins,
                    dense_ends,
                    dense_steps,
                ],
                **kwargs
            )
        ]

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls.version_10(node, **kwargs)
