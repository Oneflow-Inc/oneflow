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
from oneflow.python.framework.tensor import Tensor
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module
from typing import Optional, Sequence


@oneflow_export("Cat")
@register_op_by_module("cat")
class Cat(Module):
    """
    """

    def __init__(self, axis=0, max_dim_size: Optional[int] = None, n=2) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("concat")
            .Input("in", n)
            .Output("out")
            .Attr("axis", axis)
            .Attr("max_dim_size", max_dim_size)
            .Build()
        )

    def forward(self, inputs):
        return self._op(*inputs)[0]


# @register_op_by_module("cat")
# def concat_op(inputs):
#     return Cat(n=len(inputs))(inputs)
