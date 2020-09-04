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
import oneflow as flow

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.handler import onnx_op
from oneflow.python.onnx.handler import tf_func
from oneflow.python.ops import math_ops
from .math_mixin import ArithmeticMixin


@onnx_op("Add")
@tf_func(math_ops.add)
class Add(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return [cls.run_onnx_node(node, tensor_dict, **kwargs)]
