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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    assert dim_num >= 2
    if axis < 0:
        axis += dim_num
    assert axis >= 0
    assert axis < dim_num

    need_transpose = False
    permute = list(range(dim_num))
    if axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return need_transpose, permute


@oneflow_export("nn.ReLU")
@experimental_api
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

        import oneflow.experimental as flow
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
@experimental_api
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

        import oneflow.experimental as flow
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
@experimental_api
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
@experimental_api
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

        import oneflow.experimental as flow
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
@experimental_api
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

        import oneflow.experimental as flow
        import numpy as np
        import oneflow.typing as tp

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        input = flow.Tensor(x)
        gelu = flow.nn.GELU()
        
        out = gelu(input)
        # out [-0.15426877, 0., 0.34573123]
    """
    return GELU()(x)


@oneflow_export("nn.Sigmoid")
@experimental_api
class Sigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(
            np.array(
                [
                    [0.81733328, 0.43621480, 0.10351428],
                    [-1.15555191, -0.67776406, 0.27372134],
                ]
            )
        )
        m = flow.nn.Sigmoid() # or y = flow.sigmoid(x)
        y = m(x)
        # [[0.69366997, 0.60735673, 0.52585548],
        # [0.23947647, 0.33676055, 0.56800622]]

    """

    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("sigmoid").Input("in").Output("out").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("sigmoid")
@register_tensor_op("sigmoid")
@experimental_api
def sigmoid_op(x):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(
            np.array(
                [
                    [0.81733328, 0.43621480, 0.10351428],
                    [-1.15555191, -0.67776406, 0.27372134],
                ]
            )
        )
        y = x.sigmoid()
        # [[0.69366997, 0.60735673, 0.52585548],
        # [0.23947647, 0.33676055, 0.56800622]]

    """
    return Sigmoid()(x)


@oneflow_export("nn.Hardsigmoid")
@experimental_api
class Hardsigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3  \\
            1 & \text{ if } x \ge +3 \\
            \frac{x}{6} + \frac{1}{2} & \text{ otherwise } \\
        \end{cases}
    
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    
    For example:
    
    .. code-block:: python

        import oneflow.experimental as flow
        m = flow.nn.Hardsigmoid()
        input = flow.randn(2)
        output = m(input)
    
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        assert inplace == False, f"Hardsigmoid not support inplace equal true now!"
        self._op = flow.builtin_op("hardsigmoid").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.Softmax")
@experimental_api
class Softmax(Module):
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.axis = -1 if dim is None else dim
        self._op = flow.builtin_op("softmax").Input("in").Output("out").Build()
        self._transpose_op = (
            flow.builtin_op("transpose")
            .Input("input")
            .Output("output")
            .Attr("perm", [])
            .Build()
        )

    def forward(self, x):
        need_transpose, permute = _softmax_need_transpose(x, self.axis)
        if need_transpose:
            x = self._transpose_op(x, perm=permute)[0]

        res = self._op(x)[0]
        if need_transpose:
            res = self._transpose_op(res, perm=permute)[0]
        return res


@oneflow_export("softmax")
@register_tensor_op("softmax")
@experimental_api
def softmax_op(tensor, dim=None):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    When the input Tensor is a sparse tensor then the unspecifed
    values are treated as ``-inf``.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np

        m = flow.nn.Softmax(dim = 2)
        x = flow.Tensor(
            np.array(
                [[[[-0.46716809,  0.40112534,  0.61984003],
                [-1.31244969, -0.42528763,  1.47953856]]],

                [[[ 1.02978742, -0.49383053,  1.88214159],
                [ 1.35351622, -1.46251285, -1.40751374]]]]
            )
        )
        y = m(x)
        # [[[[0.6995764  0.6955959  0.29740235]
        # [0.3004236  0.30440408 0.7025977 ]]]

        # [[[0.4197673  0.7248568  0.96407217]
        # [0.58023274 0.27514324 0.03592779]]]]
    """
    return Softmax(dim)(tensor)


@oneflow_export("nn.LogSoftmax")
@experimental_api
class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor.
    The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        m = flow.nn.LogSoftmax(dim=1)
        x = flow.Tensor(
            np.array(
                [[ 0.4296, -1.1957,  2.5463],
                [ 1.2552, -1.5747,  0.6923]]
            )
        )
        y = m(x)
        # [[-2.251349   -3.8766491  -0.13464898]
        # [-0.48770458 -3.3176045  -1.0506046 ]]
    """

    def __init__(
        self, dim: Optional[int] = 1,
    ):
        super().__init__()
        self.dim = dim
        self._op = (
            flow.builtin_op("transpose")
            .Input("input")
            .Output("output")
            .Attr("perm", [])
            .Build()
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, x):
        need_transpose, permute = _softmax_need_transpose(x, self.dim)
        if need_transpose:
            x = self._op(x, perm=permute)[0]

        x = x.softmax()
        res = x.log()

        if need_transpose:
            res = self._op(res, perm=permute)[0]

        return res

    def extra_repr(self):
        return "dim={dim}".format(dim=self.dim)


@oneflow_export("nn.Hardtanh")
@experimental_api
class Hardtanh(Module):
    r"""
    Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        
        m = flow.nn.Hardtanh()
        arr = np.random.randn(2, 3, 4, 5)
        x = flow.Tensor(arr)
        out = m(x)
    
    """

    def __init__(
        self,
        min_val: float = -1,
        max_val: float = 1,
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        super().__init__()
        if min_value is not None:
            warnings.warn(
                "keyword argument min_value is deprecated and rename to min_val"
            )
            min_val = min_value
        if max_value is not None:
            warnings.warn(
                "keyword argument max_value is deprecated and rename to max_val"
            )
            max_val = max_value
        assert inplace == False, f"Hardtanh not support inplace equal true now!"
        self._op = (
            flow.builtin_op("hardtanh")
            .Input("in")
            .Attr("min_val", min_val)
            .Attr("max_val", max_val)
            .Output("out")
            .Build()
        )

    def forward(self, x):
        res = self._op(x)[0]
        return res
