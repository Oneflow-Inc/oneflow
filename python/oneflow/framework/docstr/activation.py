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
    oneflow.F.prelu,
    r"""
    prelu(x: Tensor, alpha: Tensor) -> Tensor  

    Applies the element-wise function:

    .. math::
        prelu(x) = max(0,x) + alpha * min(0,x) 

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.Tensor(np.asarray([[[[1, -2], [3, 4]]]]), dtype=flow.float32)
        >>> alpha = flow.nn.Parameter(flow.Tensor(1).fill_(0.25))
        >>> print(flow.F.prelu(x, alpha).numpy())
        [[[[ 1.  -0.5]
           [ 3.   4. ]]]]
   
    See
    :class:`~oneflow.nn.PReLU` for more details.
 
    """,
)


add_docstr(
    oneflow.F.gelu,
    r"""
    gelu(x: Tensor) -> Tensor 

    The equation is:

    .. math::
         out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)

        >>> out = flow.F.gelu(input)
        >>> out
        tensor([-0.1543,  0.0000,  0.3457], dtype=oneflow.float32)

    See    
    :class:`~oneflow.nn.GELU` for more details.
 
    """,
)


add_docstr(
    oneflow.F.softmax,
    r"""
    softmax(x: Tensor) -> Tensor 

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\\exp(x_i)}{\sum_j \exp(x_j)}
    
    See :class:`~oneflow.nn.Softmax` for more details.
    """,
)
add_docstr(
    oneflow.F.softplus,
    r"""
    softplus(x: Tensor) -> Tensor 

    Applies the element-wise function:

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))    
    
    See :class:`~oneflow.nn.Softplus` for more details.
    """,
)

add_docstr(
    oneflow.F.tanh,
    r"""
    tanh(x: Tensor) -> Tensor 

    The equation is:

    .. math::

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    See :class:`~oneflow.nn.Tanh` for more details.
    """,
)
add_docstr(
    oneflow.F.log_sigmoid,
    r"""
    log_sigmoid(x: Tensor) -> Tensor 

    Applies the element-wise function:

    .. math::
        \text{log_sigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)
   
    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)     
          
        >>> out = flow.F.log_sigmoid(input)
        >>> out
        tensor([-0.9741, -0.6931, -0.4741], dtype=oneflow.float32)

    See :class:`~oneflow.nn.LogSigmoid` for more details.

    """,
)

add_docstr(
    oneflow.F.softsign,
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
        >>> input = flow.Tensor(x) 
        >>> out = flow.F.softsign(input)
        >>> out
        tensor([0.5000, 0.6667, 0.7500], dtype=oneflow.float32)
 
    See :class:`~oneflow.nn.Softsign` for more details.
    
    """,
)


add_docstr(
    oneflow.F.silu,
    r"""
    silu(x: Tensor) -> Tensor

    The formula is: 

    .. math::

        \text{silu}(x) = x * sigmoid(x)
        
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)       
        >>> out = flow.F.silu(input)
        >>> out
        tensor([0.7311, 1.7616, 2.8577], dtype=oneflow.float32)

    See :class:`~oneflow.nn.SiLU` for more details.

    """,
)


add_docstr(
    oneflow.F.mish,
    r""" 
    mish(x: Tensor) -> Tensor 

    Applies the element-wise function:

    .. math::
        \text{mish}(x) = x * \text{tanh}(\text{softplus}(x))


    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)       

        >>> out = flow.F.mish(input)
        >>> out
        tensor([0.8651, 1.9440, 2.9865], dtype=oneflow.float32)

    See :class:`~oneflow.nn.Mish` for more details.
    
    """,
)


add_docstr(
    oneflow.F.relu,
    r"""
    relu(x: Tensor, inplace: bool =False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~oneflow.nn.ReLU` for more details.

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``


    """,
)
add_docstr(
    oneflow.F.hardsigmoid,
    r"""
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
    oneflow.F.hardswish,
    r"""
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
    oneflow.F.sigmoid,
    r"""
    sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~oneflow.nn.Sigmoid` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.Tensor(np.array([0.81733328, 0.43621480, 0.10351428]))
        >>> input = flow.Tensor(x)
        >>> out = flow.nn.functional.sigmoid(input)
        >>> out
        tensor([0.6937, 0.6074, 0.5259], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.hardtanh,
    r"""
    hardtanh(input, min_val=-1., max_val=1.) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~oneflow.nn.Hardtanh` for more
    details.

    """,
)
add_docstr(
    oneflow.F.leaky_relu,
    r"""
    leaky_relu(x: Tensor,  alpha :Float) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative_slope} * \min(0, x)`

    See :class:`~oneflow.nn.LeakyReLU` for more details.

    """,
)
add_docstr(
    oneflow.F.elu,
    r"""
    elu(x: Tensor, alpha :Float) -> Tensor

    Applies element-wise,
        :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`.

    See :class:`~oneflow.nn.ELU` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> out = flow.nn.functional.elu(input, alpha=1.0)
        >>> out
        tensor([-0.3935,  0.0000,  0.5000], dtype=oneflow.float32)
    """,
)
add_docstr(
    oneflow.F.selu,
    r"""
    selu(x: Tensor) -> Tensor

    Applies element-wise function :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`, with :math:`\alpha=1.6732632423543772848170429916717` and  :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~oneflow.nn.SELU` for more details.

    For examples:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> out = flow.nn.functional.selu(input)
        >>> out
        tensor([1.0507, 2.1014, 3.1521], dtype=oneflow.float32)
    """,
)
