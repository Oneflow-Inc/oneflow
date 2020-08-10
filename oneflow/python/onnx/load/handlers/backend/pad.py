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


@onnx_op("Pad")
@tf_func(tf.pad)
class Pad(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        num_dim = len(tensor_dict[node.inputs[0]].get_shape())
        mode = node.attrs.pop("mode", "constant")

        if cls.SINCE_VERSION < 11:  # for opset 1 and opset 2
            paddings = node.attrs.pop("pads", None)
            # tf requires int32 paddings
            paddings = tf.constant(
                np.transpose(np.array(paddings).reshape([2, num_dim]).astype(np.int32))
            )
            constant_values = node.attrs.pop("value", 0.0)

        else:  # for opset 11
            paddings = tensor_dict[node.inputs[1]]
            # tf requires int32 paddings
            paddings = tf.cast(
                tf.transpose(tf.reshape(paddings, [2, num_dim])), dtype=tf.int32
            )
            constant_values = (
                tensor_dict[node.inputs[2]] if len(node.inputs) == 3 else 0
            )

        def _symmetric_pad(i, x):
            paddings_i = tf.map_fn(lambda e: tf.where(i < e, 1, 0), paddings)
            paddings_i = tf.reshape(paddings_i, [num_dim, 2])
            x = tf.pad(x, paddings_i, "SYMMETRIC")
            return i + 1, x

        if mode.lower() == "edge":
            paddings = tf.reshape(paddings, [-1])
            max_i = tf.reduce_max(paddings)
            _, x = tf.while_loop(
                lambda i, x: tf.less(i, max_i),
                _symmetric_pad,
                [0, x],
                [tf.TensorShape([]), tf.TensorShape(None)],
            )
            return [x]

        return [
            cls.make_tensor_from_onnx_node(
                node, inputs=[x, paddings, mode, constant_values], **kwargs
            )
        ]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_2(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
