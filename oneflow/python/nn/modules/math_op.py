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

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import (
    register_tensor_op_by_module,
    register_op_by_module,
)


@oneflow_export("Sin")
@register_op_by_module("sin")
class Sin(Module):
    r"""
    Returns a new tensor with the sine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sin(\text{input}_{i})

    Args:
        {input}

    Keyword args:
        {out}

    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("sin").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("Cos")
@register_op_by_module("cos")
class Cos(Module):
    r"""
    Returns a new tensor with the cosine  of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    Args:
        {input}

    Keyword args:
        {out}
        
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("cos").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("Log")
@register_op_by_module("log")
class Log(Module):
    r"""
    Returns a new tensor with the natural logarithm of the elements of :attr:`input`.

    .. math::
        y_{i} = \log_{e} (x_{i})

    Args:
        {input}

    Keyword args:
        {out}
        
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("log").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]
