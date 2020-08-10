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

from oneflow.python.onnx.load.common.tf_helper import tf_shape
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("IsInf")
@tf_func(tf.math.is_inf)
class IsInf(BackendHandler):
    @classmethod
    def version_10(cls, node, **kwargs):
        inp = kwargs["tensor_dict"][node.inputs[0]]
        dtype = inp.dtype
        shape = tf_shape(inp)
        zero = tf.zeros(shape, dtype)
        dn = node.attrs.get("detect_negative", 1)
        dp = node.attrs.get("detect_positive", 1)
        # detecting only positive infinity, zero out elements < 0
        if dn == 0:
            inp = tf.maximum(zero, inp)
        # detecting only negative infinity, zero out elements > 0
        if dp == 0:
            inp = tf.minimum(zero, inp)
        return [cls.make_tensor_from_onnx_node(node, inputs=[inp], **kwargs)]
