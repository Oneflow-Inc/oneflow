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
    oneflow.F.sin,
    "\n    sin(x: Tensor) -> Tensor\n\n    Returns a new tensor with the sine of the elements of :attr:`input`.\n\n    .. math::\n\n        \\text{y}_{i} = \\sin(\\text{x}_{i})\n\n    Args:\n        x (Tensor): the input tensor.\n\n    For example:\n\n    .. code-block:: python\n\n        >>> import oneflow as flow\n        >>> import numpy as np\n        >>> x1 = flow.Tensor(np.array([-0.5461,  0.1347, -2.7266, -0.2746]).astype(np.float32))\n        >>> y1 = flow.F.sin(x1)\n        >>> y1\n        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)\n        >>> x2 = flow.Tensor(np.array([-1.4, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))\n        >>> y2 = flow.F.sin(x2)\n        >>> y2\n        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)\n\n\n",
)
add_docstr(
    oneflow.F.cos,
    "\n    cos(x: Tensor) -> Tensor\n\n    Returns a new tensor with the cosine  of the elements of :attr:`input`.\n    \n    .. math::\n        \\text{y}_{i} = \\cos(\\text{x}_{i})\n\n    Args:\n        x (Tensor): the input tensor.\n\n    For example:\n\n    .. code-block:: python\n\n        >>> import oneflow as flow\n        >>> import numpy as np\n        >>> x = np.array([1.4309,  1.2706, -0.8562,  0.9796])\n        >>> x = flow.Tensor(x, dtype=flow.float32)\n        >>> y = flow.F.cos(x)\n        >>> y\n        tensor([0.1394, 0.2957, 0.6553, 0.5574], dtype=oneflow.float32)\n\n",
)

add_docstr(
    oneflow.F.abs,
    r"""
    abs(x: Tensor) -> Tensor

    Return the absolute value of each element in input tensor:math:`y = |x|` element-wise.
    
    Args:
        input (Tensor): the input tensor.
    
    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = flow.Tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
        >>> flow.F.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.acos,
    r"""
    acos(x: Tensor) -> Tensor

    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \arccos(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.F.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.atanh,
    r"""
    atanh(x: Tensor) -> Tensor

    Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \tanh^{-1}(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.F.atanh(input)
        >>> output
        tensor([0.5493, 0.6931, 0.8673], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.broadcast_less,
    r"""
    broadcast_less(Tensor x, Tensor y) -> Tensor

    Returns the truth value of :math:`x < y` element-wise

    Args:
        x (Tensor): the input tensor.
        y (Tensor): the input tensor.
		
    Returns:
        z (Tensor): A Blob with int8 type.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.Tensor(np.array([1, 2, 4]).astype(np.float32))
        >>> z = flow.F.broadcast_less(x,y)
        >>> z
        tensor([0, 0, 1], dtype=oneflow.int8)

    """,
)

add_docstr(
    oneflow.F.broadcast_less_equal,
    r"""
    broadcast_less_equal(Tensor x, Tensor y) -> Tensor

    Returns the truth value of :math:`x <= y` element-wise.

    Args:
        x (Tensor): the input tensor.
        y (Tensor): the input tensor.
        
		
    Returns:
        z (Tensor): A tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.Tensor(np.array([1, 2, 4]).astype(np.float32))
        >>> z = flow.F.broadcast_less_equal(x,y)
        >>> z
        tensor([1, 1, 1], dtype=oneflow.int8)

    """,
)

add_docstr(
    oneflow.F.log,
    r"""
    log(Tensor x) -> Tensor

    Returns a new tensor with the natural logarithm of the elements of :attr:`input`.
    
    .. math::
        \text{out}_{i}=\log_e(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.
        	
    Returns:
        output (Tensor): the ouput tensor .

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> y = flow.F.log(x)
        >>> y
        tensor([0.2624, 0.4055, 0.9933], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.log1p,
    r"""
    log1p(Tensor x) -> Tensor

    Returns a new tensor with the natural logarithm of (1 + input).
    
    .. math::
        \text{out}_{i}=\log_e(1+\text{input}_{i})

    Args:
        input (Tensor): the input tensor.
        	
    Returns:
        output (Tensor): the ouput tensor .

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> y = flow.F.log1p(x)
        >>> y
        tensor([0.8329, 0.9163, 1.3083], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.F.sqrt,
    r"""
    sqrt(Tensor x) -> Tensor
    
    Returns a new tensor with the square-root of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \sqrt{\text{input}_{i}}

    Args:
        input (Tensor): the input tensor.

     For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> input = flow.Tensor(arr)
        >>> output = flow.sqrt(input)
        >>> output
        tensor([1.    , 1.4142, 1.7321], dtype=oneflow.float32)
        
    """,
)

add_docstr(
    oneflow.F.pow,
    r"""
    pow(Tensor x, Tensor exponent) -> Tensor

    Takes the power of each element in input with exponent and returns a tensor with the result. Exponent can be either a single float number, a single int number, or a tensor with the same shape as input.
    When exponent is a scalar value, the operation applied is:

    .. math::
        \text{out}_i = x_i ^ \text{exponent}

    When exponent is a tensor, the operation applied is:

    .. math::
        \text{out}_i = x_i ^ {\text{exponent}_i}

    Args:
        - input (Tensor): the input tensor.
        - exponent (int, float, Tensor): the exponent.

    Returns:
        Tensor: The result of variance on the specified axis of input Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        >>> y = flow.Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        >>> out = flow.pow(x, y)
        >>> out
        tensor([  1.,   4.,  27., 256.], dtype=oneflow.float32)
 
    """,
)