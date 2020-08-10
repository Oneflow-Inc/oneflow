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
from oneflow.python.onnx.load.common.tf_helper import tf_shape


@onnx_op("ArgMax")
@tf_func(tf.argmax)
class ArgMax(BackendHandler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {"default": {"axis": 0}}

    @classmethod
    def _common(cls, node, **kwargs):
        axis = node.attrs.get("axis", 0)
        keepdims = node.attrs.get("keepdims", 1)
        select_last_index = node.attrs.get("select_last_index", 0)
        if select_last_index == 0:
            arg_max = cls.make_tensor_from_onnx_node(node, **kwargs)
        else:
            # reverse the input and apply argmax on that to get last occurrence of max
            x = kwargs["tensor_dict"][node.inputs[0]]
            x = tf.reverse(x, axis=[axis])
            arg_max = cls.make_tensor_from_onnx_node(node, inputs=[x], **kwargs)
            # adjust indices to account for the reverse
            arg_max = tf_shape(x)[axis] - arg_max - 1
        if keepdims == 1:
            return [tf.expand_dims(arg_max, axis=axis)]
        return [arg_max]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_12(cls, node, **kwargs):
        return cls._common(node, **kwargs)
