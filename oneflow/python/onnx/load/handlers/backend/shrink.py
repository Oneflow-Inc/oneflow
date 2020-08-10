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


@onnx_op("Shrink")
class Shrink(BackendHandler):
    @classmethod
    def version_9(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        input_tensor = tensor_dict[node.inputs[0]]
        input_shape = tf.shape(input_tensor, out_type=tf.int64)

        # handle defaults for attributes
        lambd = node.attrs["lambd"] if "lambd" in node.attrs else 0.5
        bias = node.attrs["bias"] if "bias" in node.attrs else 0.0

        # make tensors in the right shape
        lambd_tensor = tf.fill(input_shape, tf.constant(lambd, input_tensor.dtype))
        lambd_neg_tensor = tf.fill(
            input_shape, tf.constant(lambd * -1, input_tensor.dtype)
        )
        bias_tensor = tf.fill(input_shape, tf.constant(bias, input_tensor.dtype))
        zeros_tensor = tf.zeros(input_shape, input_tensor.dtype)

        # prepare return values and conditions
        input_plus = tf.add(input_tensor, bias_tensor)
        input_minus = tf.subtract(input_tensor, bias_tensor)
        greater_cond = tf.greater(input_tensor, lambd_tensor)
        less_cond = tf.less(input_tensor, lambd_neg_tensor)

        return [
            tf.where(
                less_cond, input_plus, tf.where(greater_cond, input_minus, zeros_tensor)
            )
        ]
