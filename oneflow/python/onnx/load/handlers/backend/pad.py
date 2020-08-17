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

import oneflow.python.ops.pad as pad
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("Pad")
@tf_func(pad.pad)
class Pad(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        num_dim = len(tensor_dict[node.inputs[0]].get_shape())
        if node.attrs.get("mode", "constant") != "constant":
            raise ValueError("Pad modes other than constant are not supported.")

        paddings = tensor_dict[node.inputs[1]]
        # tf requires int32 paddings
        paddings = tf.cast(
            tf.transpose(tf.reshape(paddings, [2, num_dim])), dtype=tf.int32
        )
        constant_values = tensor_dict[node.inputs[2]] if len(node.inputs) == 3 else 0

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
