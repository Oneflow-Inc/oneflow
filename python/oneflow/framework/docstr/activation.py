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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow._C.prelu,
    """
    prelu(x: Tensor, alpha: Tensor) -> Tensor  

    Applies the element-wise function:

    .. math::
        prelu(x) = max(0,x) + alpha * min(0,x) 

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.tensor(np.asarray([[[[1, -2], [3, 4]]]]), dtype=flow.float32)
        >>> alpha = flow.nn.Parameter(flow.tensor([1], dtype=flow.float32).fill_(0.25))
        >>> flow.nn.functional.prelu(x, alpha)
        tensor([[[[ 1.0000, -0.5000],
                  [ 3.0000,  4.0000]]]], dtype=oneflow.float32,
               grad_fn=<prelu_backward>)
   
    See
    :class:`~oneflow.nn.PReLU` for more details.
 
    """,
)

add_docstr(
    oneflow.relu,
    """
    Applies the rectified linear unit function element-wise. See :class:`~oneflow.nn.ReLU` for more details. 

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    
    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> ndarr = np.asarray([1, -2, 3])
        >>> input = flow.Tensor(ndarr)
        >>> output = flow.relu(input)
        >>> output
        tensor([1., 0., 3.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.gelu,
    r"""
    gelu(x: Tensor) -> Tensor 

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
        >>> input = flow.tensor(x)

        >>> out = flow.gelu(input)
        >>> out
        tensor([-0.1543,  0.0000,  0.3457], dtype=oneflow.float32)

    See    
    :class:`~oneflow.nn.GELU` for more details.
 
    """,
)


add_docstr(
    oneflow._C.quick_gelu,
    r"""
    quick_gelu(x: Tensor) -> Tensor 

    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs

    .. math::
        \\text{QuickGELU}(x) = x * \\sigma(1.702x) = x * \\frac{1}{1 + \\exp(-1.702x)}

    Args:
        input (oneflow.Tensor): Input Tensor

    Returns:
        oneflow.Tensor: A Tensor has same shape as the input.

    See    
    :class:`~oneflow.nn.QuickGELU` for more details.
 
    """,
)

add_docstr(
    oneflow._C.softmax,
    r"""
    softmax(x: Tensor, dim: int) -> Tensor 

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\\exp(x_i)}{\sum_j \exp(x_j)}
    
    See :class:`~oneflow.nn.Softmax` for more details.
    """,
)

add_docstr(
    oneflow._C.log_softmax,
    r"""
    log_softmax(x: Tensor, dim: int) -> Tensor 

    LogSoftmax is defined as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right) = x_i - \log({ \sum_j \exp(x_j)})
    
    See :class:`~oneflow.nn.LogSoftmax` for more details.
    """,
)

add_docstr(
    oneflow._C.gumbel_softmax,
    r"""
    gumbel_softmax(x: Tensor, dim: int, tau: float = 1.0, hard: bool = False) -> Tensor 

    Solve the problem that the output values of argmax do not reflect the probability distribution of the model's output.
    Compensates for the fact that the argmax cannot participate in gradient back-propagation.

    Gumbel is defined as:

    .. math::
        Gumbel_i = -log(-log(U_i)),\ U_i \sim U(0,1)

    Add Noise ~ Gumbel:

    .. math::
        In = (In + Noise) / tau

    Calculate Softmax value:

    .. math::
        gumbel\_softmax(In)=\frac{e^{In_i/tau}}{\sum_{j=1}^n{e^{In_j/tau}}},i=1,2,3...n

    Parameters:
        x (oneflow.Tensor): the input Tensor.
        dim (int, Tuple[int]): the dimension to softmax. 
        tau (double): the input tensor of Softmax should obey the Gumbel(x, tau).
        hard (bool): if `hard=True`, the output tensor will be one-hot.
    """,
)

add_docstr(
    oneflow.softplus,
    r"""
    softplus(x: Tensor, beta: double = 1, threshold: double = 20) -> Tensor 

    Applies the element-wise function:

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))   

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`. 
    
    See :class:`~oneflow.nn.Softplus` for more details.
    """,
)

add_docstr(
    oneflow.tanh,
    r"""
    tanh(x: Tensor) -> Tensor 

    The equation is:

    .. math::

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    See :class:`~oneflow.nn.Tanh` for more details.
    """,
)
add_docstr(
    oneflow._C.logsigmoid,
    r"""
    logsigmoid(x: Tensor) -> Tensor 

    Applies the element-wise function:

    .. math::
        \text{logsigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)
   
    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.tensor(x)     
          
        >>> out = flow.nn.functional.logsigmoid(input)
        >>> out
        tensor([-0.9741, -0.6931, -0.4741], dtype=oneflow.float32)

    See :class:`~oneflow.nn.LogSigmoid` for more details.

    """,
)

add_docstr(
    oneflow._C.softsign,
    r"""
    softsign(x: Tensor) -> Tensor 

    The formula is: 
    
    .. math::  
    
        softsign(x) = \frac{x}{1 + |x|}
    
    For example:
    
    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.tensor(x) 
        >>> out = flow.nn.functional.softsign(input)
        >>> out
        tensor([0.5000, 0.6667, 0.7500], dtype=oneflow.float32)
 
    See :class:`~oneflow.nn.Softsign` for more details.
    
    """,
)


add_docstr(
    oneflow.silu,
    """
    silu(x: Tensor) -> Tensor

    The formula is: 

    .. math::

        \text{silu}(x) = x * sigmoid(x)
        
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.tensor(x)       
        >>> out = flow.silu(input)
        >>> out
        tensor([0.7311, 1.7616, 2.8577], dtype=oneflow.float32)

    See :class:`~oneflow.nn.SiLU` for more details.

    """,
)


add_docstr(
    oneflow.mish,
    """ 
    mish(x: Tensor) -> Tensor 

    Applies the element-wise function:

    .. math::
        \text{mish}(x) = x * \text{tanh}(\text{softplus}(x))


    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.tensor(x)       

        >>> out = flow.mish(input)
        >>> out
        tensor([0.8651, 1.9440, 2.9865], dtype=oneflow.float32)

    See :class:`~oneflow.nn.Mish` for more details.
    
    """,
)


add_docstr(
    oneflow._C.hardsigmoid,
    """
    hardsigmoid(x: Tensor)-> Tensor

    Applies the element-wise function

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    
    See :class:`~oneflow.nn.Hardsigmoid` for more details.
    """,
)
add_docstr(
    oneflow._C.hardswish,
    """
    hardswish(x: Tensor)-> Tensor

    Applies the hardswish function, element-wise, as described in the paper:

    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    See :class:`~oneflow.nn.Hardswish` for more details.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """,
)
add_docstr(
    oneflow.sigmoid,
    r"""
    sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~oneflow.nn.Sigmoid` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([0.81733328, 0.43621480, 0.10351428])
        >>> input = flow.tensor(x, dtype=flow.float32)
        >>> out = flow.nn.functional.sigmoid(input)
        >>> out
        tensor([0.6937, 0.6074, 0.5259], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow._C.hardtanh,
    """
    hardtanh(input, min_val=-1., max_val=1.) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~oneflow.nn.Hardtanh` for more
    details.

    """,
)
add_docstr(
    oneflow._C.leaky_relu,
    """
    leaky_relu(x: Tensor,  alpha :Float) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative_slope} * \min(0, x)`

    See :class:`~oneflow.nn.LeakyReLU` for more details.

    """,
)
add_docstr(
    oneflow._C.rrelu,
    """
    rrelu(x: Tensor, lower: Float = 1.0 / 8, upper: Float = 1.0 / 3, training: bool = False, inplace: bool = False) -> Tensor

    Applies the randomized leaky rectified liner unit function, element-wise
    :math:`\text{RReLU}(x) = \max(0, x) + a * \min(0, x)`

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.
    
    See :class:`~oneflow.nn.RReLU` for more details.

    """,
)
add_docstr(
    oneflow._C.rrelu_,
    """
    rrelu(x: Tensor, lower: Float = 1.0 / 8, upper: Float = 1.0 / 3, training: bool = False) -> Tensor

    In-place version of :func:`rrelu`.
    """,
)
add_docstr(
    oneflow._C.elu,
    """
    elu(x: Tensor, alpha :Float) -> Tensor

    Applies element-wise,
        :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`.

    See :class:`~oneflow.nn.ELU` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.tensor(x)
        >>> out = flow.nn.functional.elu(input, alpha=1.0)
        >>> out
        tensor([-0.3935,  0.0000,  0.5000], dtype=oneflow.float32)
    """,
)
add_docstr(
    oneflow.selu,
    """
    selu(x: Tensor) -> Tensor

    Applies element-wise function

    .. math::

        \text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`, with :math:`\alpha=1.6732632423543772848170429916717` and  :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~oneflow.nn.SELU` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.tensor(x)
        >>> out = flow.nn.functional.selu(input)
        >>> out
        tensor([1.0507, 2.1014, 3.1521], dtype=oneflow.float32)
    """,
)
add_docstr(
    oneflow._C.glu,
    """
    glu(input: Tensor, dim: int) -> Tensor 

    The equation is:

    .. math::
         GLU(input) = GLU(a, b) = a \otimes sigmoid(b)
    
    .. note::
        where input is split in half along dim to form a and b, ⊗ is the element-wise product between matrices.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> x = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=flow.float32)
        >>> y = nn.functional.glu(x)
        >>> y
        tensor([[0.9526, 1.9640],
                [4.9954, 5.9980]], dtype=oneflow.float32)

    See    
    :class:`~oneflow.nn.GLU` for more details.
 
    """,
)


add_docstr(
    oneflow._C.celu,
    r"""
    celu(x: Tensor, alpha: Float=1.0, inplace: bool=False) -> Tensor

    Applies the element-wise function:

    .. math::

        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    See :class:`~oneflow.nn.CELU` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.tensor(x)
        >>> out = flow.nn.functional.celu(input, alpha=0.5)
        >>> out
        tensor([-0.3161,  0.0000,  0.5000], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow._C.threshold,
    """
    threshold(input: Tensor, threshold: float, value: float) -> Tensor

    Thresholds each element of the input Tensor.

    See :class:`~oneflow.nn.Threshold` for more details.
    """,
)

add_docstr(
    oneflow._C.hardshrink,
    """
    hardshrink(input: Tensor, lambd: float=0.5, inplace: bool=False) -> Tensor

    Applies the hard shrinkage function in an element-wise manner.

    See :class:`~oneflow.nn.Hardshrink` for more details.
    """,
)

add_docstr(
    oneflow._C.softshrink,
    """
    softshrink(input: Tensor, lambd: float=0.5, inplace: bool=False) -> Tensor

    Applies the soft shrinkage function in an element-wise manner.

    See :class:`~oneflow.nn.Softshrink` for more details.
    """,
)
