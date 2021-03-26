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
from typing import (
    Optional,
    Union,
    Tuple
)


@oneflow_export("nn.Ones")
@register_tensor_op_by_module("ones")
@register_op_by_module("ones")
class Ones(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("floating_value", float(0))
            .Attr("integer_value", 1)
            .Attr("is_floating_value", False)
            .Attr("dtype", flow.int)
        )

    def forward(self, shape):
        assert shape is not None
        if shape is not None:
            assert isinstance(shape, (list, tuple))
        self._op = self._op.Attr("shape", shape).Build()
        return self._op()[0]



@oneflow_export("nn.Zeros")
@register_tensor_op_by_module("zeros")
@register_op_by_module("zeros")
class Zeros(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("floating_value", float(0.))
            .Attr("integer_value", int(0))
            .Attr("is_floating_value", True)
            .Attr("dtype", flow.float)
        )

    def forward(self, shape):
        assert shape is not None
        if shape is not None:
            assert isinstance(shape, (list, tuple))
        self._op = self._op.Attr("shape", shape).Build()
        return self._op()[0]
