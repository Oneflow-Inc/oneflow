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
from oneflow.python.ops import linalg
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op


@onnx_op("Gemm")
class Gemm(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        tensor_dict = kwargs["tensor_dict"]
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]

        if len(node.input_tensor_names) > 2:
            z = tensor_dict[node.input_tensor_names[2]]
        else:
            z = 0

        transA = False if node.attrs.get("transA", 0) == 0 else True
        transB = False if node.attrs.get("transB", 0) == 0 else True
        alpha = node.attrs.get("alpha", 1.0)
        beta = node.attrs.get("beta", 1.0)

        return [
            alpha * linalg.matmul(x, y, transpose_a=transA, transpose_b=transB)
            + beta * z
        ]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls._common(node, **kwargs)
