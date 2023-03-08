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
from typing import Optional, Tuple

import oneflow as flow
from oneflow.nn.modules.module import Module
from oneflow.framework.tensor import Tensor


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
    """Applies the element-wise function 
        :math:`\\text{ELU}(x) = \\begin{cases}x & \\text{ if } x \\gt 0  \\\\\\alpha*(exp(x)-1) & \\text{ if } x \\le 0 \\\\\\end{cases}`

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
        return flow._C.celu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self):
        param_str = f"alpha={self.alpha}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


class GELU(Module):
    """
    GELU(approximate='none') -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.GELU.html.

    Applies the Gaussian Error Linear Units function:

    .. math:: \\text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \\text{GELU}(x) = 0.5 * x * (1 + \\text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))

    Args:
        input (oneflow.Tensor): Input Tensor
        approximate (string, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Returns:
        oneflow.Tensor: A Tensor has same shape as the input.

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

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        if self.approximate == "none" or self.approximate == "tanh":
            return flow._C.gelu_with_approximate(input, self.approximate)
        else:
            raise NotImplementedError


class QuickGELU(Module):
    """
    QuickGELU() -> Tensor

    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs

    .. math::
        \\text{QuickGELU}(x) = x * \\sigma(1.702x) = x * \\frac{1}{1 + \\exp(-1.702x)}

    Args:
        input (oneflow.Tensor): Input Tensor

    Returns:
        oneflow.Tensor: A Tensor has same shape as the input.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.Tensor([-0.5, 0, 0.5])
        >>> gelu = flow.nn.QuickGELU()

        >>> out = gelu(input)
        >>> out
        tensor([-0.1496,  0.0000,  0.3504], dtype=oneflow.float32)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow._C.quick_gelu(x)


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


class Hardshrink(Module):
    r"""
    The Hardshrink activation.

    The formula is:

    .. math::
        \text{Hardshrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([-1.1, 0, 0.2, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> hardshrink = flow.nn.Hardshrink(lambd=0.5)
        >>> out = hardshrink(input)
        >>> out
        tensor([-1.1000,  0.0000,  0.0000,  0.0000], dtype=oneflow.float32)
    """

    def __init__(self, lambd: float = 0.5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.lambd = lambd

    def forward(self, x):
        return flow._C.hardshrink(x, lambd=self.lambd, inplace=self.inplace)

    def extra_repr(self) -> str:
        param_str = f"lambd={self.lambd}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


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
        return flow._C.softplus(x, beta=self.beta, threshold=self.threshold)

    def extra_repr(self):
        return f"beta={self.beta}, threshold={self.threshold}"


class Hardswish(Module):
    """Applies the hardswish function, element-wise, as described in the paper `Searching for MobileNetV3
    <https://arxiv.org/abs/1905.02244>`__.

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
        return flow._C.leaky_relu(x, alpha=self.negative_slope, inplace=self.inplace)

    def extra_repr(self):
        param_str = f"negative_slope={self.negative_slope}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


class RReLU(Module):
    """Applies the randomized leaky rectified liner unit function, element-wise:

    .. math::
        \\text{RReLU}(x) = \\begin{cases}
            x, & \\text{ if } x \\geq 0 \\\\
            a \\times x, & \\text{ otherwise }
        \\end{cases}
        
    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.
    
    .. note::
        See `Empirical Evaluation of Rectified Activations in Convolution Network: <https://arxiv.org/pdf/1505.00853.pdf>`_

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> m = flow.nn.RReLU(0.1, 0.3)
        >>> arr = np.array([0.2, -0.3, -3.0, 4.0, 0.5, -2.2])
        >>> x = flow.Tensor(arr)
        >>> out = m(x) 
        >>> print(out) # doctest: +SKIP
        tensor([ 0.2000, -0.0824, -0.5418,  4.0000,  0.5000, -0.4213], dtype=oneflow.float32) # doctest: +SKIP
            
    """

    def __init__(
        self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, x):
        return flow._C.rrelu(x, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        param_str = f"lower={self.lower}"
        param_str += f"upper={self.upper}"
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


class Softshrink(Module):
    r"""
    The Softshrink activation.

    The formula is:
    
    .. math::

        \text{Softshrink}(x) =
        \begin{cases}
        x - \lambd, & \text{ if } x > \lambda \\
        x + \lambd, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.Softshrink.html.

    Args:
        lambd: the :math:`\lambda` value for the Softshrink formulation. Default: 0.5
        inplace: can optionally do the operation in-place. Default: ``False``
    
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example:
    
    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([-1, 0, 0.2, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> softshrink = flow.nn.Softshrink(lambd=0.5)
        >>> out = softshrink(input)
        >>> out
        tensor([-0.5000,  0.0000,  0.0000,  0.0000], dtype=oneflow.float32)
    """

    def __init__(self, lambd: float = 0.5, inplace: bool = False):
        self.inplace = inplace
        self.lambd = lambd
        super().__init__()

    def forward(self, x):
        return flow._C.softshrink(x, alpha=self.lambd, inplace=self.inplace)

    def extra_repr(self) -> str:
        param_str = f"lambd={self.lambd}"
        param_str += ", inplace=True" if self.inplace else ""
        return param_str


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


class Threshold(Module):
    r"""The Threshold Activation. Return ``x`` if ``x`` is greater than ``threshold``, else return ``value``.

    The interface is consistent with PyTorch.
    The documentation is referenced from https://pytorch.org/docs/1.10/generated/torch.nn.Threshold.html.

    The formula is:

    .. math::

        \text{Threshold}(x) =
        \begin{cases}
        x, & \text{ if } x > \text{ threshold } \\
        \text{value }, & \text{ otherwise }
        \end{cases}

    Args:
        threshold (float): The ``threshold`` value for the Threshold formulation
        value (float): The ``value`` value for the Threshold formulation

    Shapes:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Returns:
        Oneflow.Tensor: The result tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([-1, 0, 0.5, 1]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> th = flow.nn.Threshold(threshold=0.5, value=0.2)
        >>> out = th(input)
        >>> out
        tensor([0.2000, 0.2000, 0.2000, 1.0000], dtype=oneflow.float32)

    """

    def __init__(self, threshold: float, value: float):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, input):
        return flow._C.threshold(input, threshold=self.threshold, value=self.value)


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    The interface is consistent with PyTorch.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.MultiheadAttention.html

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ["batch_first"]
    bias_k: Optional[Tensor]
    bias_v: Optional[Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = flow.nn.Parameter(
                flow.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = flow.nn.Parameter(
                flow.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = flow.nn.Parameter(
                flow.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = flow.nn.Parameter(
                flow.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = flow.nn.Parameter(
                flow.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = flow.nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = flow.nn.Parameter(
                flow.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_v = flow.nn.Parameter(
                flow.empty((1, 1, embed_dim), **factory_kwargs)
            )
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            flow.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            flow.nn.init.xavier_uniform_(self.q_proj_weight)
            flow.nn.init.xavier_uniform_(self.k_proj_weight)
            flow.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            flow.nn.init.constant_(self.in_proj_bias, 0.0)
            flow.nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            flow.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            flow.nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif (
            self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype
        ):
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif flow.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            if not all([(x.is_cuda or "cpu" in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif flow.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                return flow._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1
                    if key_padding_mask is not None
                    else 0
                    if attn_mask is not None
                    else None,
                )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            (
                attn_output,
                attn_output_weights,
            ) = flow.nn.functional.functional_multi_head_attention.multi_head_attention(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
            )
        else:
            (
                attn_output,
                attn_output_weights,
            ) = flow.nn.functional.functional_multi_head_attention.multi_head_attention(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
