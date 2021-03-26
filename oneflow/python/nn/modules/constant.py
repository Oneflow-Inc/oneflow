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
from typing import Optional, Tuple


@oneflow_export("nn.Ones")
class Ones(Module):
    def __init__(self, dtype: Optional[flow.dtype] = None) -> None:
        super().__init__()
        if dtype==None or dtype==flow.int:
            dtype = flow.int
            floating_value = float(0)
            integer_value = int(1)
            is_floating_value = False
        else:
            dtype = flow.float
            floating_value = float(1)
            integer_value = int(0)
            is_floating_value = True

        self._op = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("floating_value", floating_value)
            .Attr("integer_value", integer_value)
            .Attr("is_floating_value", is_floating_value)
            .Attr("dtype", dtype)
        )

    def forward(self, shape):
        assert shape is not None, "shape should not be None!"
        assert isinstance(shape, (int, list, tuple)), "shape should be int, list or tuple format!"
        if isinstance(shape, (int)):
            shape = [shape]
        self._op = self._op.Attr("shape", shape).Build()
        return self._op()[0]



@oneflow_export("nn.Zeros")
class Zeros(Module):
    def __init__(self, dtype: Optional[flow.dtype] = None) -> None:
        super().__init__()
        if dtype==None or dtype==flow.float:
            dtype = flow.float
            floating_value = float(0.)
            integer_value = int(0)
            is_floating_value = True
        else:
            dtype = flow.int
            floating_value = float(0)
            integer_value = int(0)
            is_floating_value = False

        self._op = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("floating_value", floating_value)
            .Attr("integer_value", integer_value)
            .Attr("is_floating_value", is_floating_value)
            .Attr("dtype", dtype)
        )

    def forward(self, shape):
        assert shape is not None, "shape should not be None!"
        assert isinstance(shape, (int, list, tuple)), "shape should be int, list or tuple format!"
        if isinstance(shape, (int)):
            shape = [shape]
        self._op = self._op.Attr("shape", shape).Build()
        return self._op()[0]
