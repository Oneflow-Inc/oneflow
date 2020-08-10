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


@onnx_op("QuantizeLinear")
class QuantizeLinear(BackendHandler):
    @classmethod
    def version_10(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        y_scale = tensor_dict[node.inputs[1]]

        x = tf.cast(x, tf.float32)
        y = tf.divide(x, y_scale)
        y = tf.round(y)
        if len(node.inputs) == 3:
            y_zero_point = tensor_dict[node.inputs[2]]
            y_dtype = y_zero_point.dtype
            y_zero_point = tf.cast(y_zero_point, tf.float32)
            y = tf.add(y, y_zero_point)
        else:  # y_zero_point default dtype = uint8
            y_dtype = tf.uint8

        y = tf.saturate_cast(y, y_dtype)

        return [y]
