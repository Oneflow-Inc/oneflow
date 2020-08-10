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
from .math_mixin import ReductionMixin


@onnx_op("ReduceSumSquare")
class ReduceSumSquare(ReductionMixin, BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        axis = node.attrs.get("axes", list(range(len(x.get_shape().as_list()))))
        keepdims = node.attrs.get("keepdims", 1) == 1
        return [tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims)]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
