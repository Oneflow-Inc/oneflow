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
    oneflow.add,
    r"""Computes the addition of `input` by `other` for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input + other

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise add
        >>> x = flow.Tensor(np.random.randn(2,3))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # scalar add
        >>> x = 5
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # broadcast add
        >>> x = flow.Tensor(np.random.randn(1,1))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """
)

add_docstr(
    oneflow.div,
    r"""Computes the division of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = \\frac{input}{other}
    
    Args:
        input (Union[int, float, flow.Tensor]): input.
        other (Union[int, float, flow.Tensor]): other.
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise divide
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar divide
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast divide
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape 
        (2, 3)

    """
)

add_docstr(
    oneflow.mul,
    r"""Computes the multiplication of input by other for each element, scalar and broadcast promotation are supported.
    
    The formula is:

    .. math::
        out = input \\times other
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise multiply
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar mutiply
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast mutiply
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape 
        (2, 3)

    """
)

add_docstr(
    oneflow.reciprocal,
    r"""Computes the safe reciprocal of x. If x is zero, the reciprocal will
    be also set to zero.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> out = flow.reciprocal(x)
        >>> out.numpy()
        array([[1.        , 0.5       , 0.33333334],
               [0.25      , 0.2       , 0.16666667]], dtype=float32)
    """
)

add_docstr(
    oneflow.sub,
    r"""Computes the subtraction of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input - other
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise subtract
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar subtract
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast subtract
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

    """
)

add_docstr(
    oneflow.asin,
    r"""
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sin^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([-0.5,  0.8, 1.0,  -0.8]), dtype=flow.float32)
        >>> output = flow.asin(input)
        >>> output.shape
        flow.Size([4])
        >>> output
        tensor([-0.5236,  0.9273,  1.5708, -0.9273], dtype=oneflow.float32)
        >>> input1 = flow.Tensor(np.array([[0.8, 1.0], [-0.6, -1.0]]), dtype=flow.float32)
        >>> output1 = input1.asin()
        >>> output1.shape
        flow.Size([2, 2])
        >>> output1
        tensor([[ 0.9273,  1.5708],
                [-0.6435, -1.5708]], dtype=oneflow.float32)
    """
)

add_docstr(
    oneflow.asinh,
    r"""
    Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sinh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([2, 3, 4]), dtype=flow.float32)
        >>> output = flow.asinh(input)
        >>> output.shape
        flow.Size([3])
        >>> output
        tensor([1.4436, 1.8184, 2.0947], dtype=oneflow.float32)

        >>> input1 = flow.Tensor(np.array([[-1, 0, -0.4], [5, 7, 0.8]]), dtype=flow.float32)
        >>> output1 = input1.asinh()
        >>> output1.shape
        flow.Size([2, 3])
        >>> output1
        tensor([[-0.8814,  0.0000, -0.3900],
                [ 2.3124,  2.6441,  0.7327]], dtype=oneflow.float32)

    """
)

add_docstr(
    oneflow.atan,
    r"""
    Returns a new tensor with the arctangent of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tan^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([0.5, 0.6, 0.7]), dtype=flow.float32)
        >>> output = flow.atan(input)
        >>> output.shape
        flow.Size([3])
        
    """
)

add_docstr(
    oneflow.log1p,
    r"""Returns a new tensor with the natural logarithm of (1 + input).

    .. math::
        \\text{out}_{i}=\\log_e(1+\\text{input}_{i})

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> out = flow.log1p(x)
        >>> out
        tensor([0.8329, 0.9163, 1.3083], dtype=oneflow.float32)

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
