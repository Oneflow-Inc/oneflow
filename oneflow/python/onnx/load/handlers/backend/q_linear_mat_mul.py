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


@onnx_op("QLinearMatMul")
class QLinearMatMul(BackendHandler):
    @classmethod
    def version_10(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        a = tensor_dict[node.inputs[0]]
        a_scale = tensor_dict[node.inputs[1]]
        a_zero_point = tensor_dict[node.inputs[2]]
        b = tensor_dict[node.inputs[3]]
        b_scale = tensor_dict[node.inputs[4]]
        b_zero_point = tensor_dict[node.inputs[5]]
        y_scale = tensor_dict[node.inputs[6]]
        y_zero_point = tensor_dict[node.inputs[7]]
        y_dtype = y_zero_point.dtype

        # reshape 1-D a_scale, a_zero_point, y_scale and
        # y_zero_point so it can broadcast in arithmetic
        # operations later
        a_scale_shape = a_scale.get_shape().as_list()
        if a_scale_shape and a_scale_shape[0] > 1:
            a_scale = tf.reshape(a_scale, [a_scale_shape[0], 1])
            a_zero_point = tf.reshape(a_zero_point, [a_scale_shape[0], 1])
        y_scale_shape = y_scale.get_shape().as_list()
        if y_scale_shape and y_scale_shape[0] > 1:
            y_scale = tf.reshape(y_scale, [y_scale_shape[0], 1])
            y_zero_point = tf.reshape(y_zero_point, [y_scale_shape[0], 1])

        # cast all inputs to float32
        a = tf.cast(a, tf.float32)
        a_zero_point = tf.cast(a_zero_point, tf.float32)
        b = tf.cast(b, tf.float32)
        b_zero_point = tf.cast(b_zero_point, tf.float32)
        y_zero_point = tf.cast(y_zero_point, tf.float32)

        # dequantize a and b
        dequantized_a = tf.subtract(a, a_zero_point)
        dequantized_a = tf.multiply(dequantized_a, a_scale)
        dequantized_b = tf.subtract(b, b_zero_point)
        dequantized_b = tf.multiply(dequantized_b, b_scale)

        # matmul
        x = tf.matmul(dequantized_a, dequantized_b)

        # quantize x
        y = tf.divide(x, y_scale)
        y = tf.round(y)
        y = tf.add(y, y_zero_point)
        y = tf.saturate_cast(y, y_dtype)

        return [y]
