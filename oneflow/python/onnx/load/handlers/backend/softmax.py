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


@onnx_op("Softmax")
@tf_func(tf.nn.softmax)
class Softmax(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        axis = node.attrs.get("axis", 1)
        axis = axis if axis >= 0 else len(np.shape(x)) + axis

        if axis == len(np.shape(x)) - 1:
            return [cls.make_tensor_from_onnx_node(node, **kwargs)]

        shape = tf.shape(x)
        cal_shape = (
            tf.reduce_prod(shape[0:axis]),
            tf.reduce_prod(shape[axis : tf.size(shape)]),
        )
        x = tf.reshape(x, cal_shape)

        return [tf.reshape(tf.nn.softmax(x), shape)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
