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
import oneflow._oneflow_internal
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module


@oneflow_export("nn.Sigmoid")
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("sigmoid").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.ReLU")
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("relu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.Tanh")
@register_op_by_module("tanh")
class Tanh(Module):
    r"""This operator computes the hyperbolic tangent value of Tensor.

    The equation is: 

    .. math:: 

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    Args:
        x (oneflow.Tensor): A Tensor
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow.Tensor: The result Tensor

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np

        x = np.array([-1, 0, 1]).astype(np.float32)
        input = flow.Tensor(x)
        tanh = flow.nn.Tanh()
        out = tanh(input).numpy()

        # out [-0.7615942  0.         0.7615942]

    """

    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("tanh").Input("x").Output("y").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@register_tensor_op_by_module("tanh")
def tanh_op():
    return Tanh()(tensor)


@oneflow_export("nn.GeLU")
@register_op_by_module("gelu")
class GeLU(Module):
    r"""Gelu activation operator.

    The equation is:

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    Args:
        x (oneflow.Tensor): Input Tensor
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        input = flow.Tensor(x)
        gelu = flow.nn.GeLU()
        
        out = gelu(input)

        # out [-0.15426877, 0., 0.34573123]

    """

    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("gelu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@register_tensor_op_by_module("gelu")
def gelu_op(tensor):
    return GeLU()(tensor)
