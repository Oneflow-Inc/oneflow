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
from oneflow.python.onnx.load.handlers.handler import partial_support
from oneflow.python.onnx.load.handlers.handler import ps_description


@onnx_op("Cast")
@tf_func(tf.cast)
@partial_support(True)
@ps_description(
    "Cast string to float32/float64/int32/int64 " + "are not supported in Tensorflow."
)
class Cast(BackendHandler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {"rename": {"to": "dtype"}}

    @classmethod
    def version_1(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    @classmethod
    def version_6(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    @classmethod
    def version_9(cls, node, **kwargs):
        inp = kwargs["tensor_dict"][node.inputs[0]]
        to_type = node.attrs.get("to")

        if to_type == tf.string:
            return [tf.as_string(inp)]

        if inp.dtype == tf.string:
            if to_type not in [tf.float32, tf.float64, tf.int32, tf.int64]:
                raise RuntimeError(
                    "Cast string to type {} is not supported in Tensorflow.".format(
                        to_type
                    )
                )
            return [tf.strings.to_number(inp, to_type)]

        return [cls.make_tensor_from_onnx_node(node, **kwargs)]
