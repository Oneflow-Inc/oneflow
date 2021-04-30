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
from oneflow.python.framework.tensor import register_tensor_op


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
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        m = flow.nn.ReLU()
        arr = np.random.randn(2, 3, 4, 5)
        input = flow.Tensor(arr)
        output = m(input)
        # equal to np.maximum(0, arr)

    """

    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("relu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.Tanh")
class Tanh(Module):
    r"""This operator computes the hyperbolic tangent value of Tensor.

    The equation is:

    .. math::

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    Args:
        x (oneflow.Tensor): A Tensor

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


@oneflow_export("tanh")
@register_tensor_op("tanh")
def tanh_op(x):
    r"""This operator computes the hyperbolic tangent value of Tensor.

    The equation is: 

    .. math:: 

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    Args:
        x (oneflow.Tensor): A Tensor

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
    return Tanh()(x)


@oneflow_export("nn.GELU")
class GELU(Module):
    r"""Gelu activation operator.

    The equation is:

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    Args:
        x (oneflow.Tensor): Input Tensor

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        input = flow.Tensor(x)
        gelu = flow.nn.GELU()

        out = gelu(input)

        # out [-0.15426877, 0., 0.34573123]

    """

    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("gelu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("gelu")
@register_tensor_op("gelu")
def gelu_op(x):
    r"""Gelu activation operator.

    The equation is:

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    Args:
        x (oneflow.Tensor): Input Tensor

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        input = flow.Tensor(x)
        gelu = flow.nn.GELU()
        
        out = gelu(input)

        # out [-0.15426877, 0., 0.34573123]

    """
    return GELU()(x)
