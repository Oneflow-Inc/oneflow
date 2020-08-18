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
from oneflow.python.ops import nn_ops
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op
from oneflow.python.onnx.load.handlers.handler import tf_func


@onnx_op("SoftmaxCrossEntropyLoss")
@tf_func(nn_ops.sparse_softmax_cross_entropy_with_logits)
class SoftmaxCrossEntropyLoss(BackendHandler):
    @classmethod
    def version_12(cls, node, **kwargs):
        tensor_dict = kwargs.get("tensor_dict", {})
        inputs = (
            tensor_dict[node.input_tensor_names[1]],
            tensor_dict[node.input_tensor_names[0]],
        )
        return [cls.make_tensor_from_onnx_node(node, inputs=inputs, **kwargs)]
