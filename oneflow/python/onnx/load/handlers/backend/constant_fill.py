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


@onnx_op("ConstantFill")
@tf_func(tf.fill)
class ConstantFill(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        if node.inputs and "shape" in node.attrs:
            raise ValueError(
                "Cannot set the shape argument and pass in an input at the same time."
            )
        if not node.inputs and "extra_shape" in node.attrs:
            raise ValueError("Cannot set extra_shape when there is no input.")

    @classmethod
    def get_attrs_processor_param(cls):
        return {"default": {"value": 0.0}}

    @classmethod
    def version_1(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]

        if "shape" in node.attrs:
            shape = node.attrs["shape"]
        else:
            shape = (
                tensor_dict[node.inputs[0]].get_shape().as_list()
                if node.attrs.get("input_as_shape", 0) == 0
                else tensor_dict[node.inputs[0]]
            )

        if "extra_shape" in node.attrs:
            shape = tf.concat([shape, node.attrs["extra_shape"]], 0)

        value = node.attrs.get("value", 0.0)

        if "dtype" in node.attrs:
            return [tf.cast(tf.fill(shape, value), dtype=node.attrs["dtype"])]

        return [cls.make_tensor_from_onnx_node(node, inputs=[shape], **kwargs)]
