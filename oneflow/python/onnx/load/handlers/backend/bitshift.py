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


@onnx_op("BitShift")
class BitShift(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        tf_func = (
            tf.bitwise.left_shift
            if node.attrs.get("direction") == "LEFT"
            else tf.bitwise.right_shift
        )
        return [cls.make_tensor_from_onnx_node(node, tf_func=tf_func, **kwargs)]

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
