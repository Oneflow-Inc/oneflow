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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module
import oneflow.python.framework.id_util as id_util
from typing import Optional, Sequence


@oneflow_export("MatMul")
@register_tensor_op_by_module("tmp.matmul")
@register_op_by_module("tmp.matmul")
class MatMul(Module):
    r"""
    """

    def __init__(name: Optional[str] = None,) -> None:
        super().__init__()

        self._op = (
            flow.builtin_op("matmul")
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", False)
            .Attr("transpose_b", True)
            .Attr("alpha", 1.0)
            .Build()
        )

    def forward(self, a, b):
        return self._op(a, b)[0]
