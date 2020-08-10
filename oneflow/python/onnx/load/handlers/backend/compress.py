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
import copy
import tensorflow as tf

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("Compress")
@tf_func(tf.gather)
class Compress(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        attrs = copy.deepcopy(node.attrs)
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        condition = tensor_dict[node.inputs[1]]

        x = tf.reshape(x, [-1]) if node.attrs.get("axis") is None else x
        if condition.shape.is_fully_defined():
            condition_shape = condition.shape[0]
            indices = tf.constant(list(range(condition_shape)), dtype=tf.int64)
        else:
            condition_shape = tf.shape(condition, out_type=tf.int64)[0]
            indices = tf.range(condition_shape, dtype=tf.int64)
        not_zero = tf.not_equal(condition, tf.zeros_like(condition))
        attrs["indices"] = tf.boolean_mask(indices, not_zero)
        return [cls.make_tensor_from_onnx_node(node, inputs=[x], attrs=attrs, **kwargs)]

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
