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
from .conv_mixin import ConvMixin


@onnx_op("QLinearConv")
class QLinearConv(ConvMixin, BackendHandler):
    @classmethod
    def _dequantize_tensor(cls, base, zero_point, scale):
        # Do computation in float32
        base = tf.cast(base, tf.float32)
        zero_point = tf.cast(zero_point, tf.float32)
        return (base - zero_point) * scale

    @classmethod
    def _dequantize_w(cls, base, zero_point, scale):
        tensor_list = [
            cls._dequantize_tensor(base[i][j], zero_point[j], scale[j])
            for i in range(base.shape.as_list()[0])
            for j in range(zero_point.shape.as_list()[0])
        ]

        out_tensor = tf.concat(tensor_list, 0)
        return tf.reshape(out_tensor, base.shape)

    @classmethod
    def version_10(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        x_scale = tensor_dict[node.inputs[1]]
        x_zero_point = tensor_dict[node.inputs[2]]
        w = tensor_dict[node.inputs[3]]
        w_scale = tensor_dict[node.inputs[4]]
        w_zero_point = tensor_dict[node.inputs[5]]
        y_scale = tensor_dict[node.inputs[6]]
        y_zero_point = tensor_dict[node.inputs[7]]

        output_dtype = x.dtype

        # Convert w_zero_point and w_scale to 1-D if scalar
        if len(w_zero_point.shape) == 0:
            w_zero_point = tf.fill([x.shape[1]], w_zero_point)
        elif len(w_zero_point.shape) > 1:
            raise ValueError("Unsupported zero point: {}".format(w_zero_point))

        if len(w_scale.shape) == 0:
            w_scale = tf.fill([x.shape[1]], w_scale)
        elif len(w_scale.shape) > 1:
            raise ValueError("Unsupported scale: {}".format(w_scale))

        # Dequantize variables to float32
        x = cls._dequantize_tensor(x, x_zero_point, x_scale)
        w = cls._dequantize_w(w, w_zero_point, w_scale)
        y_zero_point = tf.cast(y_zero_point, tf.float32)

        new_dict = tensor_dict.copy()
        new_dict[node.inputs[0]] = x
        new_dict[node.inputs[3]] = w

        # if bias is defined save it here
        B = (
            tensor_dict[node.inputs[8]]
            if len(node.inputs) == 9
            else tf.constant([0], tf.float32)
        )
        if len(node.inputs) == 9:
            B = tf.cast(B, tf.float32)
            B_scale = x_scale * w_scale
            B = tf.round(B / B_scale)
            # Remore bias from inputs
            node.inputs.remove(node.inputs[8])

        # Remove scales and zero-points from inputs
        for i in [7, 6, 5, 4, 2, 1]:
            node.inputs.remove(node.inputs[i])

        # Use common conv handling
        conv_node = cls.conv(node, new_dict)[0]

        # Process output
        y = tf.round(conv_node / y_scale) + y_zero_point

        # Add bias to the convolution
        y = y + B

        return [tf.cast(y, output_dtype)]
