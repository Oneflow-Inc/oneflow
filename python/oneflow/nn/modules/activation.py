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
import warnings
from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_inplace_valid


class PReLU(Module):
    """Applies the element-wise function:

    .. math::
        PReLU(x) = \\max(0,x) + a * \\min(0,x)

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
        >>> import oneflow as flow
        
        >>> m = flow.nn.PReLU()
        >>> input = flow.tensor(np.asarray([[[[1, -2], [3, 4]]]]), dtype=flow.float32)
        >>> print(m(input).numpy())
        [[[[ 1.  -0.5]
           [ 3.   4. ]]]]

    """

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = flow.nn.Parameter(
            flow.empty(num_parameters, dtype=dtype, device=device).fill_(init)
        )

    def forward(self, x):
        return flow._C.prelu(x, self.weight)

    def extra_repr(self) -> str:
        return "num_parameters={}".format(self.num_parameters)


class ReLU(Module):
    """Applies the rectified linear unit function element-wise:

    :math:`\\text{ReLU}(x) = (x)^+ = \\max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> relu = flow.nn.ReLU()
        >>> ndarr = np.asarray([1, -2, 3])
        >>> x = flow.Tensor(ndarr)
        >>> relu(x)
        tensor([1., 0., 3.], dtype=oneflow.float32)

    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            _check_inplace_valid(x)
        return flow._C.relu(x, self.inplace)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class ReLU6(Module):
    """Applies the element-wise function:

    .. math::

        \\text{Relu6}(x) = \\begin{cases}
            6 & \\text{ if } x > 6 \\\\
            0 & \\text{ if } x < 0 \\\\
            x & \\text{ otherwise } \\\\
        \\end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> relu6 = flow.nn.ReLU6()

        >>> out = relu6(input)
        >>> out
        tensor([0.0000, 0.0000, 0.5000], dtype=oneflow.float32)

    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            warnings.warn("ReLU6 module do not support inplace now")
        return flow._C.hardtanh(x, min_val=0.0, max_val=6.0)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


def relu6(input, inplace=False):
    r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~oneflow.nn.ReLU6` for more details.
    """
    if inplace:
        warnings.warn("nn.functional.relu6 do not support inplace now")
    return flow._C.hardtanh(input, min_val=0.0, max_val=6.0)


class Tanh(Module):
    """This operator computes the hyperbolic tangent value of Tensor.

    The equation is:

    .. math::

        out = \\frac{e^x-e^{-x}}{e^x+e^{-x}}

    Args:
        input (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-1, 0, 1]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> tanh = flow.nn.Tanh()
        >>> out = tanh(input)
        >>> out
        tensor([-0.7616,  0.0000,  0.7616], dtype=oneflow.float32)

    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return flow._C.tanh(input)


class ELU(Module):
    """Applies the element-wise function:

    .. math::

        \\text{ELU}(x) = \\begin{cases}
				x & \\text{ if } x \\gt 0  \\\\
                \\alpha*(exp(x)-1) & \\text{ if } x \\le 0 \\\\
    		    \\end{cases}

    Args:
        alpha: the :math:`\\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> elu = flow.nn.ELU()

        >>> out = elu(input)
        >>> out
        tensor([-0.3935,  0.0000,  0.5000], dtype=oneflow.float32)

    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            warnings.warn("ELU module do not support inplace now")
        return flow._C.elu(x, alpha=self.alpha)

    def extra_repr(self):
        param_str = f"alpha={self.alpha}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


class CELU(Module):
    """Applies the element-wise function:

    .. math::

        \\text{CELU}(x, \\alpha) = \\begin{cases}
				x & \\text{ if } x \\ge 0  \\\\
                \\alpha*(exp(\\frac{x}{\\alpha})-1) & \\text{ otherwise } \\\\
    		    \\end{cases}

    Args:
        alpha: the :math:`\\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> celu = flow.nn.CELU(alpha=0.5)

        >>> out = celu(input)
        >>> out
        tensor([-0.3161,  0.0000,  0.5000], dtype=oneflow.float32)

    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            _check_inplace_valid(x)
        return flow._C.celu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self):
        param_str = f"alpha={self.alpha}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


class GELU(Module):
    """Gelu activation operator.

    The equation is:

    .. math::
        out = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    Args:
        x (oneflow.Tensor): Input Tensor

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> gelu = flow.nn.GELU()

        >>> out = gelu(input)
        >>> out
        tensor([-0.1543,  0.0000,  0.3457], dtype=oneflow.float32)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow._C.gelu(x)


class Sigmoid(Module):
    """Applies the element-wise function:

    .. math::
        \\text{Sigmoid}(x) = \\sigma(x) = \\frac{1}{1 + \\exp(-x)}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.array([0.81733328, 0.43621480, 0.10351428]))
        >>> m = flow.nn.Sigmoid()
        >>> out = m(x)
        >>> out
        tensor([0.6937, 0.6074, 0.5259], dtype=oneflow.float32)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow._C.sigmoid(x)


class Hardsigmoid(Module):
    """Applies the element-wise function:

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{ if } x \\le -3  \\\\
            1 & \\text{ if } x \\ge +3 \\\\
            \\frac{x}{6} + \\frac{1}{2} & \\text{ otherwise } \\\\
        \\end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> hardsigmoid = flow.nn.Hardsigmoid()

        >>> out = hardsigmoid(input)
        >>> out
        tensor([0.4167, 0.5000, 0.5833], dtype=oneflow.float32)


    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return flow._C.hardsigmoid(x, True)
        return flow._C.hardsigmoid(x, False)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}

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
        >>> import oneflow as flow
        
        >>> m = flow.nn.Softmax(dim = 2)
        >>> x = flow.Tensor(
        ...    np.array(
        ...        [[[-0.46716809,  0.40112534,  0.61984003],
        ...        [-1.31244969, -0.42528763,  1.47953856]]]
        ...    )
        ... )
        >>> out = m(x)
        >>> out
        tensor([[[0.1575, 0.3754, 0.4671],
                 [0.0507, 0.1230, 0.8263]]], dtype=oneflow.float32)
    """

    def __init__(self, dim: Optional[int] = None):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return flow._C.softmax(x, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class LogSoftmax(Module):
    r"""Applies the LogSoftmax function to an n-dimensional
    input Tensor.
    The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right) = x_i - \log({ \sum_j \exp(x_j)})

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.LogSoftmax(dim=1)
        >>> x = flow.Tensor(
        ...    np.array(
        ...        [[ 0.4296, -1.1957,  2.5463],
        ...        [ 1.2552, -1.5747,  0.6923]]
        ...    )
        ... )
        >>> out = m(x)
        >>> out
        tensor([[-2.2513, -3.8766, -0.1346],
                [-0.4877, -3.3176, -1.0506]], dtype=oneflow.float32)
    """

    def __init__(self, dim: Optional[int] = None):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return flow._C.log_softmax(x, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class LogSigmoid(Module):
    """Applies the element-wise function:

    .. math::
        \\text{LogSigmoid}(x) = \\log\\left(\\frac{ 1 }{ 1 + \\exp(-x)}\\right)

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> logsigmoid = flow.nn.LogSigmoid()

        >>> out = logsigmoid(input)
        >>> out
        tensor([-0.9741, -0.6931, -0.4741], dtype=oneflow.float32)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow._C.logsigmoid(x)


class Softplus(Module):
    """Applies the element-wise function:

    .. math::
        \\text{Softplus}(x) = \\frac{1}{\\beta} * \\log(1 + \\exp(\\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \\times \\beta > threshold`.

    Args:
        beta: the :math:`\\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> softplus = flow.nn.Softplus()

        >>> out = softplus(input)
        >>> out
        tensor([0.4741, 0.6931, 0.9741], dtype=oneflow.float32)
    """

    def __init__(self, beta: int = 1, threshold: int = 20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        return flow.where(
            x * self.beta > self.threshold,
            x,
            1 / self.beta * flow.log(1.0 + flow.exp(self.beta * x)),
        )

    def extra_repr(self):
        return f"beta={self.beta}, threshold={self.threshold}"


class Hardswish(Module):
    """Applies the hardswish function, element-wise, as described in the paper:
    `Searching for MobileNetV3`_.

    .. math::
        \\text{Hardswish}(x) = \\begin{cases}
            0 & \\text{ if } x \\le -3  \\\\
            x & \\text{ if } x \\ge +3 \\\\
            x*(x+3)/6 & \\text{ otherwise } \\\\
        \\end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> hardswish = flow.nn.Hardswish()

        >>> out = hardswish(input)
        >>> out
        tensor([-0.2083,  0.0000,  0.2917], dtype=oneflow.float32)

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            warnings.warn("Hardswish module do not support inplace now")
        return flow._C.hardswish(x)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Hardtanh(Module):
    """
    Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \\text{HardTanh}(x) = \\begin{cases}
            1 & \\text{ if } x > 1 \\\\
            -1 & \\text{ if } x < -1 \\\\
            x & \\text{ otherwise } \\\\
        \\end{cases}

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
        >>> import oneflow as flow
        
        >>> m = flow.nn.Hardtanh()
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 1.0000, 1.0000], dtype=oneflow.float32)

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
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            warnings.warn("Hardtanh module do not support inplace now")
        return flow._C.hardtanh(x, min_val=self.min_val, max_val=self.max_val)

    def extra_repr(self):
        param_str = f"min_val={self.min_val}, max_val={self.max_val}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


class LeakyReLU(Module):
    """Applies the element-wise function:

    .. math::
        \\text{LeakyRELU}(x) = \\begin{cases}
            x, & \\text{ if } x \\geq 0 \\\\
            \\text{negative_slope} \\times x, & \\text{ otherwise }
        \\end{cases}

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
        >>> import oneflow as flow
        
        >>> m = flow.nn.LeakyReLU(0.1)
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 3.0000, 4.0000], dtype=oneflow.float32)
    """

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            warnings.warn("LeakyReLU module do not support inplace now")
        return flow._C.leaky_relu(x, alpha=self.negative_slope)

    def extra_repr(self):
        param_str = f"negative_slope={self.negative_slope}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


class Mish(Module):
    """Applies the element-wise function:

    .. math::
        \\text{Mish}(x) = x * \\text{Tanh}(\\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> mish = flow.nn.Mish()

        >>> out = mish(input)
        >>> out
        tensor([0.8651, 1.9440, 2.9865], dtype=oneflow.float32)
    """

    def __init__(self, inplace: bool = False):
        self.inplace = inplace
        super().__init__()

    def forward(self, x):
        return flow._C.mish(x)


class SiLU(Module):
    r"""SiLU(Swish) activation:

    .. math::
    
        \text{SiLU}(x) = x * sigmoid(x)
    
    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    
    For example:
    
    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow


        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> silu = flow.nn.SiLU()
        >>> out = silu(input)
        >>> out
        tensor([0.7311, 1.7616, 2.8577], dtype=oneflow.float32)
    """

    def __init__(self, inplace: bool = False):
        self.inplace = inplace
        super().__init__()

    def forward(self, x):
        return flow._C.silu(x)


class SELU(Module):
    r"""Applies the element-wise function:

    The formula is: 
    
    .. math::  
    
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))
    
    with :math:`\alpha = 1.6732632423543772848170429916717` and
    
    :math:`\text{scale} = 1.0507009873554804934193349852946`.
    
    .. warning::
    
        When using ``kaiming_normal`` or ``kaiming_normal_`` for initialisation,
        ``nonlinearity='linear'`` should be used instead of ``nonlinearity='selu'``
        in order to get `Self-Normalizing Neural Networks`_.
        See :func:`torch.nn.init.calculate_gain` for more information.
    
    More details can be found in the paper `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_.
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    
    For example:
    
    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> selu = flow.nn.SELU()
        >>> out = selu(input)
        >>> out
        tensor([1.0507, 2.1014, 3.1521], dtype=oneflow.float32)
    """

    def __init__(self, inplace: bool = False):
        self.inplace = inplace
        super().__init__()

    def forward(self, x):
        return flow._C.selu(x)


class Softsign(Module):
    r"""The SoftSign activation.

    The formula is: 
    
    .. math::  
    
        SoftSign(x) = \frac{x}{1 + |x|}
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    
    For example:
    
    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> softsign = flow.nn.Softsign()
        >>> out = softsign(input)
        >>> out
        tensor([0.5000, 0.6667, 0.7500], dtype=oneflow.float32)
    """

    def __init__(self, inplace: bool = False):
        self.inplace = inplace
        super().__init__()

    def forward(self, x):
        return flow._C.softsign(x)


class GLU(Module):
    r"""The GLU activation.

    Args:
        input (Tensor, float): input tensor. 
        dim (int, optional): dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    The formula is: 
    
    .. math::  

        GLU(input) = GLU(a, b) = a \otimes sigmoid(b)

    .. note::
        where input is split in half along dim to form a and b, ⊗ is the element-wise product between matrices.

    For example:
    
    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> m = nn.GLU()
        >>> x = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=flow.float32)
        >>> y = m(x)
        >>> y
        tensor([[0.9526, 1.9640],
                [4.9954, 5.9980]], dtype=oneflow.float32)
    
    """

    def __init__(self, dim: Optional[int] = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return flow._C.glu(input, self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
