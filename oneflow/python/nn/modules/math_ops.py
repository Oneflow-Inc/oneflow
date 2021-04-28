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
from oneflow.python.framework.tensor import register_tensor_op


class Sin(Module):
    r"""
    Returns a new tensor with the sine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sin(\text{input}_{i})

    Args:
        input (Tensor) – the input tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        arr = np.array([-0.5461,  0.1347, -2.7266, -0.2746])
        input = flow.Tensor(arr, dtype=flow.float32)
        output = flow.sin(input)

        # output
        # [-0.51935846  0.13429303 -0.40318328 -0.27116194]

    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("sin").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("sin")
@register_tensor_op("sin")
def sin_op(tensor):
    return Sin()(tensor)


class Cos(Module):
    r"""
    Returns a new tensor with the cosine  of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    Args:
        input (Tensor) – the input tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        arr = np.array([1.4309,  1.2706, -0.8562,  0.9796])
        input = flow.Tensor(arr, dtype=flow.float32)
        output = flow.cos(input)

        # output
        # [0.13944048 0.29570782 0.6553126  0.5573547 ]
        
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("cos").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("cos")
@register_tensor_op("cos")
def cos_op(tensor):
    return Cos()(tensor)


class Log(Module):
    r"""
    Returns a new tensor with the natural logarithm of the elements of :attr:`input`.

    .. math::
        y_{i} = \log_{e} (x_{i})

    Args:
        input (Tensor) – the input tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        arr = np.random.randn(2, 3, 4, 5)
        input = flow.Tensor(arr, dtype=flow.float32)
        output = flow.log(input)
        
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("log").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("log")
@register_tensor_op("log")
def log_op(tensor):
    return Log()(tensor)
