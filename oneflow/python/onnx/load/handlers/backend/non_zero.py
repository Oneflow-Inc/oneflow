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


@onnx_op("NonZero")
class NonZero(BackendHandler):
    @classmethod
    def version_9(cls, node, **kwargs):
        input_tensor = kwargs["tensor_dict"][node.inputs[0]]
        condition = tf.not_equal(input_tensor, tf.zeros_like(input_tensor))
        nonzero_indices = tf.where(condition)
        return [tf.transpose(nonzero_indices)]
