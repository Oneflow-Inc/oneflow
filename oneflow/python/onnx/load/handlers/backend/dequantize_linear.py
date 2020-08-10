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


@onnx_op("DequantizeLinear")
class DequantizeLinear(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        if len(node.inputs) == 3:
            x = tensor_dict[node.inputs[0]]
            x_scale = tensor_dict[node.inputs[1]]
            x_zero_point = tensor_dict[node.inputs[2]]
            if x_scale.shape != x_zero_point.shape:
                raise ValueError(
                    "DequantizeLinear x_scale(shape="
                    + str(x_scale.shape)
                    + ") and x_zero_point(shape="
                    + str(x_zero_point.shape)
                    + ") must be in the same shape"
                )
            if x_zero_point.dtype != x.dtype:
                raise ValueError(
                    "DequantizeLinear x_zero_point("
                    + str(x_zero_point.dtype)
                    + ") and x("
                    + str(x.dtype)
                    + ") must be in the same dtype"
                )

    @classmethod
    def version_10(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        x = tf.cast(x, tf.float32)
        x_scale = tensor_dict[node.inputs[1]]
        if len(node.inputs) == 3 and x.dtype != tf.int32:
            x_zero_point = tensor_dict[node.inputs[2]]
            x_zero_point = tf.cast(x_zero_point, tf.float32)
            x = tf.subtract(x, x_zero_point)

        y = tf.multiply(x, x_scale)

        return [y]
