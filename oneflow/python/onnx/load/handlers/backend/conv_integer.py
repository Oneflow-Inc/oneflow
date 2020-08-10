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


@onnx_op("ConvInteger")
class ConvInteger(ConvMixin, BackendHandler):
    @classmethod
    def _apply_zero_point(cls, base, zero_point):
        base = tf.cast(base, tf.float32)
        zero_point = tf.cast(zero_point, tf.float32)
        return base - zero_point

    @classmethod
    def version_10(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        w = tensor_dict[node.inputs[1]]

        def process_conv(new_x, new_w):
            # Remove zero-points from inputs
            if len(node.inputs) == 4:
                node.inputs.remove(node.inputs[3])
            if len(node.inputs) == 3:
                node.inputs.remove(node.inputs[2])

            new_dict = {node.inputs[0]: new_x, node.inputs[1]: new_w}

            # Use common conv handling
            conv_node = cls.conv(node, new_dict)

            return conv_node

        # Apply x_zero_point first
        x = (
            cls._apply_zero_point(x, tensor_dict[node.inputs[2]])
            if len(node.inputs) > 2
            else tf.cast(x, tf.float32)
        )

        # Apply w_zero_point next
        if len(node.inputs) == 4:
            w_zero_point = tensor_dict[node.inputs[3]]
            if w_zero_point.shape.rank == 0:
                # Simply apply w_zero_point for scalar
                w = cls._apply_zero_point(w, w_zero_point)
            elif w_zero_point.shape.rank == 1:
                # Need additional processing for 1d w_zero_point
                tensor_list = []
                process_shape = [1] + [w.shape[i] for i in range(1, len(w.shape))]
                for i in range(w.shape.as_list()[0]):
                    # Apply w_zero_point for each element in 1d tensor
                    out_tensor = cls._apply_zero_point(w[i], w_zero_point[i])
                    tensor_list.append(tf.reshape(out_tensor, process_shape))
                w = tf.concat(tensor_list, 0)
            else:
                raise ValueError("Unsupported w zero point: {}".format(w_zero_point))
        else:
            # Just cast without processing w
            w = tf.cast(w, tf.float32)

        return [tf.cast(process_conv(x, w)[0], tf.int32)]
