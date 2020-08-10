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

from oneflow.python.onnx.load.common import exception
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func
from oneflow.python.onnx.load.handlers.handler import partial_support
from oneflow.python.onnx.load.handlers.handler import ps_description
from .control_flow_mixin import ComparisonMixin


@onnx_op("Equal")
@tf_func(tf.equal)
@partial_support(True)
@ps_description(
    "Equal inputs in uint16/uint32/uint64 " + "are not supported in Tensorflow."
)
class Equal(ComparisonMixin, BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        supported_dtype = [
            tf.bfloat16,
            tf.half,
            tf.float32,
            tf.float64,
            tf.uint8,
            tf.int8,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.complex64,
            tf.quint8,
            tf.qint8,
            tf.qint32,
            tf.string,
            tf.bool,
            tf.complex128,
        ]
        x = kwargs["tensor_dict"][node.inputs[0]]
        if x.dtype not in supported_dtype:
            exception.OP_UNSUPPORTED_EXCEPT(
                "Equal inputs in " + str(x.dtype) + " which", "Tensorflow"
            )

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.limited_broadcast(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    @classmethod
    def version_11(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]
