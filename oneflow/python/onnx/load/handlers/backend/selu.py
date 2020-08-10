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
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("Selu")
@tf_func(tf.nn.selu)
class Selu(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        if "alpha" not in node.attrs and "gamma" not in node.attrs:
            return [cls.make_tensor_from_onnx_node(node, **kwargs)]

        x = tensor_dict[node.inputs[0]]
        alpha = node.attrs.get("alpha", 1.67326319217681884765625)
        gamma = node.attrs.get("gamma", 1.05070102214813232421875)

        return [
            tf.clip_by_value(x, 0, tf.reduce_max(x)) * gamma
            + (tf.exp(tf.clip_by_value(x, tf.reduce_min(x), 0)) - 1) * alpha * gamma
        ]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)
