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
    oneflow.abs,
    r"""

    Return the absolute value of each element in input tensor:math:`y = |x|` element-wise.

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.Tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
        >>> flow.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)
    
    """
)


add_docstr(
    oneflow.exp,
    r"""

    This operator computes the exponential of Tensor.

    The equation is:

    .. math::

        out = e^x

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.exp(x)
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

    """
)

add_docstr(
    oneflow.acos,
    r"""
    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\arccos(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)
    """
)

add_docstr(
    oneflow.acosh,
    r"""
    Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

    .. math::

        \\text{out}_{i} = \\cosh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> out1 = flow.acosh(x1)
        >>> out1
        tensor([1.3170, 1.7627, 2.0634], dtype=oneflow.float32)
        >>> x2 = flow.Tensor(np.array([1.5, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.acosh(x2)
        >>> out2
        tensor([0.9624, 1.6094, 1.9827], device='cuda:0', dtype=oneflow.float32)
    """
)

add_docstr(
    oneflow.atanh,
    r"""Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tanh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.atanh(input)
        >>> output
        tensor([0.5493, 0.6931, 0.8673], dtype=oneflow.float32)

    """
)

add_docstr(
    oneflow.sign,
    r"""Computes the sign of Tensor.

    .. math::

        \\text{out}_{i}  = \\text{sgn}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([-2, 0, 2]).astype(np.float32))
        >>> out1 = flow.sign(x1)
        >>> out1.numpy()
        array([-1.,  0.,  1.], dtype=float32)
        >>> x2 = flow.Tensor(np.array([-3.2, -4.5, 5.8]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.sign(x2)
        >>> out2.numpy()
        array([-1., -1.,  1.], dtype=float32)

    """
)

add_docstr(
    oneflow.sinh,
    r"""Returns a new tensor with the hyperbolic sine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sinh(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x1 = flow.Tensor(np.array([1, 2, 3]))
        >>> x2 = flow.Tensor(np.array([1.53123589,0.54242598,0.15117185]))
        >>> x3 = flow.Tensor(np.array([1,0,-1]))

        >>> flow.sinh(x1).numpy()
        array([ 1.1752012,  3.6268604, 10.017875 ], dtype=float32)
        >>> flow.sinh(x2).numpy()
        array([2.20381  , 0.5694193, 0.1517483], dtype=float32)
        >>> flow.sinh(x3).numpy()
        array([ 1.1752012,  0.       , -1.1752012], dtype=float32)

    """
)

add_docstr(
    oneflow.tan,
    r"""Returns  the tan value of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tan(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([-1/4*np.pi, 0, 1/4*np.pi]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.tan(input)
        >>> output
        tensor([-1.,  0.,  1.], dtype=oneflow.float32)

    """
)

add_docstr(
    oneflow._C.sin,
    r"""
    sin(x: Tensor) -> Tensor

    Returns a new tensor with the sine of the elements of :attr:`input`.

    .. math::

        \text{y}_{i} = \sin(\text{x}_{i})

    Args:
        x (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x1 = flow.Tensor(np.array([-0.5461,  0.1347, -2.7266, -0.2746]).astype(np.float32))
        >>> y1 = flow._C.sin(x1)
        >>> y1
        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)
        >>> x2 = flow.Tensor(np.array([-1.4, 2.6, 3.7]).astype(np.float32), device=flow.device('cuda'))
        >>> y2 = flow._C.sin(x2)
        >>> y2
        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)
    """,
)
add_docstr(
    oneflow._C.cos,
    r"""
    cos(x: Tensor) -> Tensor

    Returns a new tensor with the cosine  of the elements of :attr:`input`.
    
    .. math::
        \text{y}_{i} = \cos(\text{x}_{i})

    Args:
        x (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = np.array([1.4309,  1.2706, -0.8562,  0.9796])
        >>> x = flow.Tensor(x, dtype=flow.float32)
        >>> y = flow._C.cos(x)
        >>> y
        tensor([0.1394, 0.2957, 0.6553, 0.5574], dtype=oneflow.float32)
    
    """,
)
