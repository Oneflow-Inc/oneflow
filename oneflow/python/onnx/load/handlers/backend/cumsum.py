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
from oneflow.python.onnx.load.handlers.handler import partial_support
from oneflow.python.onnx.load.handlers.handler import ps_description
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("CumSum")
@tf_func(tf.math.cumsum)
@partial_support(True)
@ps_description("CumSum inputs in uint32/uint64 " + "are not supported in Tensorflow.")
class CumSum(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        supported_dtype = [
            tf.bfloat16,
            tf.half,
            tf.float32,
            tf.float64,
            tf.uint8,
            tf.uint16,
            tf.int8,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.complex64,
            tf.complex128,
        ]
        x = kwargs["tensor_dict"][node.inputs[0]]
        if x.dtype not in supported_dtype:
            exception.OP_UNSUPPORTED_EXCEPT(
                "CumSum input in " + str(x.dtype) + " which", "Tensorflow"
            )

    @classmethod
    def version_11(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.inputs[0]]
        inputs = [x]

        if len(node.inputs) > 1:
            # optional 0-D tensor, range [-rank(x), rank(x)-1]
            axis = tensor_dict["axis"]
            inputs.append(axis)

        attrs = {
            "exclusive": bool(node.attrs.get("exclusive", 0)),
            "reverse": bool(node.attrs.get("reverse", 0)),
        }

        return [cls.make_tensor_from_onnx_node(node, inputs=inputs, attrs=attrs)]
