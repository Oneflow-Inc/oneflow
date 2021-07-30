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
    oneflow.F.floor,
    r"""
    oneflow.F.floor(x: Tensor) -> Tensor
    The floor function takes a input tensor x, and outputs the greatest integer(s) less than or equal to x, that is

    .. math::

        \lfloor x\rfloor=\max \{m \in \mathbb{Z} \mid m \leq x\}

    Args:
        x(tensor, dtype=flow.float32): the input real number
        output(tensor, dtype=flow.float32)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor(15.37)
        >>> x1 = flow.F.floor(x)
        >>> x1
        tensor([15.], dtype=oneflow.float32)

    Note that x can be a non-single-elemented tensor, for example

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([2.3,3,3,8.4])
        >>> x1 = flow.F.floor(x)
        >>> x1
        tensor([2., 3., 3., 8.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.F.broadcast_not_equal,
    r"""
    oneflow.F.broadcast_not_equal(x: Tensor, y: Tensor) -> z: Tensor

    Returns a boolean Tensor z, which is resulted by contrasting Tensor x and Tensor y

    Args:

        x (Tensor): the input tensor.
        y (Tensor): the input tensor.

    Returns:
        
        z (Tensor, dtype = flow.int8): a output boolean tensor
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1,2,3])
        >>> y = flow.tensor([3,4,3])
        >>> z = flow.F.broadcast_not_equal(x, y)
        >>> z
        tensor([1, 1, 0], dtype=oneflow.int8)
    """,
)

add_docstr(
    oneflow.F.square,
    r"""
    oneflow.F.square(x: Tensor) -> y: Tensor

    Returns the element-wise squared result of the input tensor x

    Args:

        x (Tensor, dtype=flow.float32): the input tensor
    
    Returns:

        y (Tensor, dtype=flow.float32): the output tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> y = flow.square(x)
        >>> y
        tensor([1., 4., 9.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.F.broadcast_sub,
    r"""
    oneflow.F.broadcast_sub (x: Tensor, y: Tensor) -> z: Tensor

    Returns the result of Tensor x subtracting Tensor y

    Args:

        x (Tensor): the input tensor being subtracted
        y (Tensor): the input tensor that subtracts x

    Returns

        z (Tensor): the output tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([4,9,3])
        >>> y = flow.tensor([2,3,8])
        >>> z = flow.F.broadcast_sub(x, y)
        >>> z
        tensor([ 2,  6, -5], dtype=oneflow.int64)

    Note that x and y do not necessarily needs to be the same shape as long as one is broadcastable to the other, for example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[1,2,3],[4,5,6]])
        >>> y = flow.tensor([7,8,9])
        >>> z = flow.F.broadcast_sub(x, y)
        >>> z
        tensor([[-6, -6, -6],
                [-3, -3, -3]], dtype=oneflow.int64)
    """,
)
