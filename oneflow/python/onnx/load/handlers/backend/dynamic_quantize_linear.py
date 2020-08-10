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


@onnx_op("DynamicQuantizeLinear")
class DynamicQuantizeLinear(BackendHandler):
    @classmethod
    def version_11(cls, node, **kwargs):
        # A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.

        # Scale is calculated as:
        #   y_scale = (max(x) - min(x))/(qmax - qmin)
        # Zero point is calculated as:
        #   intermediate_zero_point = qmin - min(x)/y_scale
        #   y_zero_point = cast(round(saturate(intermediate_zero_point)))
        # Data quantization formula is:
        #   y = saturate(round(x / y_scale) + y_zero_point)
        # Only uint8 is supported, so saturation range is [0, 255]

        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        dtype = tf.uint8
        qmin = dtype.min
        qmax = dtype.max
        min_x = tf.math.minimum(0.0, tf.math.reduce_min(x))
        max_x = tf.math.maximum(0.0, tf.math.reduce_max(x))
        y_scale = (max_x - min_x) / (qmax - qmin)
        intermediate_zero_point = qmin - (min_x / y_scale)
        y_zero_point = tf.clip_by_value(tf.round(intermediate_zero_point), qmin, qmax)
        y = tf.cast(
            tf.clip_by_value((tf.round(x / y_scale) + y_zero_point), qmin, qmax), dtype
        )

        return [y, y_scale, tf.cast(y_zero_point, dtype)]
