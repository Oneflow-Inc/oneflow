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
import copy

import tensorflow as tf

from onnx import numpy_helper

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("ConstantOfShape")
@tf_func(tf.fill)
class ConstantOfShape(BackendHandler):
    @classmethod
    def version_9(cls, node, **kwargs):
        attrs = copy.deepcopy(node.attrs)

        shape = kwargs["tensor_dict"][node.inputs[0]]

        # make sure the shape dtype is either int32 or int64
        if shape.dtype not in [tf.int64, tf.int32]:
            shape = tf.cast(shape, tf.int64)

        # the default value is 0, float32
        if "value" in node.attrs:
            attr_value = node.attrs["value"]
            value = numpy_helper.to_array(attr_value)
            attrs["value"] = value[0]
        else:
            attrs["value"] = 0.0

        return [
            cls.make_tensor_from_onnx_node(node, inputs=[shape], attrs=attrs, **kwargs)
        ]
