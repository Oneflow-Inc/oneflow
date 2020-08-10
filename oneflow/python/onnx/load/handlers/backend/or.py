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
from .control_flow_mixin import LogicalMixin


@onnx_op("Or")
@tf_func(tf.logical_or)
class Or(LogicalMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.limited_broadcast(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]
