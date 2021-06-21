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
    if dim_num == 1:
        return False, None
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


@oneflow_export("nn.PReLU")
@experimental_api
class PReLU(Module):
    """Applies the element-wise function:

    .. math::
        PReLU(x) = \max(0,x) + a * \min(0,x)

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Attr:
        - weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.PReLU()
        >>> input = flow.Tensor(np.asarray([[[[1, -2], [3, 4]]]]), dtype=flow.float32)
        >>> print(m(input).numpy())
        [[[[ 1.  -0.5]
           [ 3.   4. ]]]]

    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = flow.nn.Parameter(flow.Tensor(num_parameters, 1, 1).fill_(init))
        self.op = flow.builtin_op("prelu").Input("x").Input("alpha").Output("y").Build()

    def forward(self, x):
        assert (
            self.num_parameters == 1 or self.num_parameters == x.shape[1]
        ), f"num_parameters in prelu must be 1 or {x.shape[1]}"
        return self.op(x, self.weight)[0]


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

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> relu = flow.nn.ReLU()
        >>> ndarr = np.asarray([1, -2, 3])
        >>> x = flow.Tensor(ndarr)
        >>> relu(x).numpy()
        array([1., 0., 3.], dtype=float32)

    """

    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, x):
        return flow.F.relu(x)


@oneflow_export("nn.ReLU6")
@experimental_api
class ReLU6(Module):
    r"""Applies the element-wise function:

    .. math::

        \text{Relu6}(x) = \begin{cases}
            6 & \text{ if } x > 6 \\
            0 & \text{ if } x < 0 \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> relu6 = flow.nn.ReLU6()

        >>> out = relu6(input).numpy()
        >>> print(out)
        [0.  0.  0.5]

    """

    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, x):
        return flow.F.hardtanh(x, min_val=0.0, max_val=6.0)


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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-1, 0, 1]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> tanh = flow.nn.Tanh()
        >>> out = tanh(input).numpy()
        >>> print(out)
        [-0.7615942  0.         0.7615942]

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.tanh(x)


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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-1, 0, 1]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> tanh = flow.nn.Tanh()
        >>> out = tanh(input).numpy()
        >>> print(out)
        [-0.7615942  0.         0.7615942]

    """
    return Tanh()(x)


@oneflow_export("nn.ELU")
@experimental_api
class ELU(Module):
    r"""Applies the element-wise function:

    .. math::

        \text{ELU}(x) = \begin{cases}
				x & \text{ if } x \gt 0  \\
                \alpha*(exp(x)-1) & \text{ if } x \le 0 \\
    		    \end{cases}

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> elu = flow.nn.ELU()

        >>> out = elu(input).numpy()
        >>> print(out)
        [-0.39346933  0.          0.5       ]

    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return flow.F.elu(x, alpha=self.alpha)


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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> gelu = flow.nn.GELU()

        >>> out = gelu(input).numpy()
        >>> print(out)
        [-0.15426877  0.          0.34573123]

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.gelu(x)


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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> gelu = flow.nn.GELU()

        >>> out = gelu(input).numpy()
        >>> print(out)
        [-0.15426877  0.          0.34573123]

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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.array([0.81733328, 0.43621480, 0.10351428]))
        >>> m = flow.nn.Sigmoid()
        >>> out = m(x).numpy()
        >>> print(out)
        [0.69367   0.6073567 0.5258555]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.sigmoid(x)


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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.array([0.81733328, 0.43621480, 0.10351428]))
        >>> out = flow.sigmoid(x).numpy()
        >>> print(out)
        [0.69367   0.6073567 0.5258555]

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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> hardsigmoid = flow.nn.Hardsigmoid()

        >>> out = hardsigmoid(input).numpy()
        >>> print(out)
        [0.41666666 0.5        0.5833333 ]


    """

    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, x):
        return flow.F.hardsigmoid(x)


@oneflow_export("nn.Softmax")
@experimental_api
class Softmax(Module):
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.axis = -1 if dim is None else dim

    def forward(self, x):
        need_transpose, permute = _softmax_need_transpose(x, self.axis)
        if need_transpose:
            x = flow.F.transpose(x, perm=permute)

        res = flow.F.softmax(x)
        if need_transpose:
            res = flow.F.transpose(res, perm=permute)
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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.Softmax(dim = 2)
        >>> x = flow.Tensor(
        ...    np.array(
        ...        [[[-0.46716809,  0.40112534,  0.61984003],
        ...        [-1.31244969, -0.42528763,  1.47953856]]]
        ...    )
        ... )
        >>> out = m(x).numpy()
        >>> print(out)
        [[[0.15752424 0.3753552  0.46712062]
          [0.05065432 0.12300029 0.8263454 ]]]
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

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.LogSoftmax(dim=1)
        >>> x = flow.Tensor(
        ...    np.array(
        ...        [[ 0.4296, -1.1957,  2.5463],
        ...        [ 1.2552, -1.5747,  0.6923]]
        ...    )
        ... )
        >>> out = m(x).numpy()
        >>> print(out)
        [[-2.2513487 -3.8766491 -0.1346489]
         [-0.4877046 -3.3176045 -1.0506046]]
    """

    def __init__(
        self, dim: Optional[int] = 1,
    ):
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, x):
        need_transpose, permute = _softmax_need_transpose(x, self.dim)
        if need_transpose:
            x = flow.F.transpose(x, perm=permute)

        x = x.softmax()
        res = x.log()

        if need_transpose:
            res = flow.F.transpose(res, perm=permute)

        return res

    def extra_repr(self):
        return "dim={dim}".format(dim=self.dim)


@oneflow_export("nn.LogSigmoid")
@experimental_api
class LogSigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> logsigmoid = flow.nn.LogSigmoid()

        >>> out = logsigmoid(input).numpy()
        >>> print(out)
        [-0.974077   -0.6931472  -0.47407696]

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        sigmoid_res = flow.experimental.sigmoid(x)
        res = flow.experimental.log(sigmoid_res)
        return res


@oneflow_export("nn.Softplus")
@experimental_api
class Softplus(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> softplus = flow.nn.Softplus()

        >>> out = softplus(input).numpy()
        >>> print(out)
        [0.474077  0.6931472 0.974077 ]
    """

    def __init__(self, beta: int = 1, threshold: int = 20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        return flow.experimental.where(
            x * self.beta > self.threshold,
            x,
            1
            / self.beta
            * flow.experimental.log(1.0 + flow.experimental.exp(self.beta * x)),
        )


@oneflow_export("nn.Hardswish")
@experimental_api
class Hardswish(Module):
    r"""Applies the hardswish function, element-wise, as described in the paper:
    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{ if } x \le -3  \\
            x & \text{ if } x \ge +3 \\
            x*(x+3)/6 & \text{ otherwise } \\
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> hardswish = flow.nn.Hardswish()

        >>> out = hardswish(input).numpy()
        >>> print(out)
        [-0.20833333  0.          0.29166666]

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, x):
        return flow.F.hardswish(x)


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


        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.Hardtanh()
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x).numpy()
        >>> print(out)
        [0.2 0.3 1.  1. ]

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

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return flow.F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


@oneflow_export("nn.LeakyReLU")
@experimental_api
class LeakyReLU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyRELU}(x) = \begin{cases}
            x, & \text{ if } x \geq 0 \\
            \text{negative_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.LeakyReLU(0.1)
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x).numpy()
        >>> print(out)
        [0.2 0.3 3.  4. ]
    """

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return flow.F.leaky_relu(x, alpha=self.negative_slope)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
