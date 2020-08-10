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
import tensorflow as tf

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("TopK")
@tf_func(tf.nn.top_k)
class TopK(BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_rank = len(x.get_shape())
        axes = list(range(x_rank))
        axis = node.attrs.get("axis", -1)
        axis = axis if axis >= 0 else axis + x_rank

        if axis != x_rank - 1:
            pre_perm = [a for a in axes if a != axis] + [axis]
            post_perm = axes[:axis] + [x_rank - 1] + axes[axis : x_rank - 1]
            x = tf.transpose(x, perm=pre_perm)
            values, indices = tf.nn.top_k(x, k=node.attrs["k"])
            values = tf.transpose(values, perm=post_perm)
            return [values, tf.cast(indices, dtype=tf.int64)]

        values, indices = tf.nn.top_k(x, k=node.attrs["k"])
        return [values, tf.cast(indices, dtype=tf.int64)]

    @classmethod
    def version_10(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_rank = len(x.get_shape())
        axes = list(range(x_rank))
        axis = node.attrs.get("axis", -1)
        axis = axis if axis >= 0 else axis + x_rank
        k = kwargs["tensor_dict"][node.inputs[1]][0]
        k = tf.cast(k, dtype=tf.int32)

        if axis != x_rank - 1:
            pre_perm = [a for a in axes if a != axis] + [axis]
            post_perm = axes[:axis] + [x_rank - 1] + axes[axis : x_rank - 1]
            x = tf.transpose(x, perm=pre_perm)
            values, indices = tf.nn.top_k(x, k)
            values = tf.transpose(values, perm=post_perm)
            return [values, tf.cast(indices, dtype=tf.int64)]

        values, indices = tf.nn.top_k(x, k)
        return [values, tf.cast(indices, dtype=tf.int64)]

    @classmethod
    def version_11(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        x_rank = len(x.get_shape())
        axes = list(range(x_rank))
        axis = node.attrs.get("axis", -1)
        axis = axis if axis >= 0 else axis + x_rank
        largest = node.attrs.get("largest", 1)
        sort = node.attrs.get("sorted", 1)
        sort = False if sort == 0 else True
        k = kwargs["tensor_dict"][node.inputs[1]][0]
        k = tf.cast(k, dtype=tf.int32)

        if largest == 0:
            x = tf.negative(x)

        if axis != x_rank - 1:
            pre_perm = [a for a in axes if a != axis] + [axis]
            post_perm = axes[:axis] + [x_rank - 1] + axes[axis : x_rank - 1]
            x = tf.transpose(x, perm=pre_perm)
            values, indices = tf.nn.top_k(x, k, sort)
            values = tf.transpose(values, perm=post_perm)
        else:
            values, indices = tf.nn.top_k(x, k, sort)

        if largest == 0:
            values = tf.negative(values)

        return [values, tf.cast(indices, dtype=tf.int64)]
