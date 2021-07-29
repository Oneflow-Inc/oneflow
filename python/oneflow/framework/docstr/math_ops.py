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
    oneflow.F.exp,
    r"""
    exp(x: Tensor) -> Tensor

    Function implementation of :func:`oneflow.Tensor.exp`.
    
    Args:
        x (Tensor): the input tensor.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.F.exp(x)
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

""",
)

add_docstr(
    oneflow.F.batch_matmul,
    r"""
    batch_matmul(a: Tensor, b: Tensor, *, transpose_a=False: Bool, transpose_b=False: Bool, alpha=1.0: Double) -> Tensor 

    Performs a batch matrix-matrix product of matrices stored in :attr:`a` and :attr:`b`.

    :attr:`a` and :attr:`b` must be 3-D tensors each containing the same number of matrices.

    If :attr:`a` is a (b x n x m) tensor, :attr:`b` is a (b x m x p) tensor, out will be a (b x n x p) tensor.
    
    Args:
        a (Tensor): the first batch of matrices to be multiplied.
        b (Tensor): the second batch of matrices to be multiplied.
        transpose_a (Bool): whether to transpose the last two dimensions on :attr:`a`.
        transpose_b (Bool): whether to transpose the last two dimensions on :attr:`b`.
        alpha (Double): the scaling factor on matrix-matrix product.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.random.randn(10, 3, 4), dtype=flow.float32)
        >>> y = flow.Tensor(np.random.randn(10, 4, 5), dtype=flow.float32)
        >>> res = flow.F.batch_matmul(x, y)
        >>> res.shape
        flow.Size([10, 3, 5])

""",
)

add_docstr(
    oneflow.F.bias_add,
    r"""
    bias_add(x: Tensor, bias: Tensor, *, axis=1: Int32) -> Tensor

    Adds :attr:`bias` to the :attr:`axis` of the tensor :attr:`x` and returns a new resulting tensor.
    
    Args:
        x (Tensor): the input tensor.
        bias (Tensor): the bias tensor.
        axis (Int32): the axis to perform addition operation.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.random.randn(10, 3, 5), dtype=flow.float32)
        >>> y = flow.Tensor(np.random.randn(5), dtype=flow.float32)
        >>> res = flow.F.bias_add(x, y, axis=2)
        >>> res.shape
        flow.Size([10, 3, 5])

""",
)

add_docstr(
    oneflow.F.broadcast_equal,
    r"""
    broadcast_equal(x: Tensor, y: Tensor) -> Tensor

    Computes :attr:`x` = :attr:`y` element-wise.

    :attr:`y` should be a tensor whose shape is broadcastable with tensor :attr:`x`.
    
    Args:
        x (Tensor): the tensor to compare.
        y (Tensor): the tensor to compare.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> flow.F.broadcast_equal(flow.tensor([1, 2]), flow.tensor([1, 2]))
        tensor([1, 1], dtype=oneflow.int8)
        >>> flow.F.broadcast_equal(flow.tensor([1, 2]), flow.tensor([1]))
        tensor([1, 0], dtype=oneflow.int8)

""",
)

add_docstr(
    oneflow.F.broadcast_greater_equal,
    r"""
    broadcast_greater_equal(x: Tensor, y: Tensor) -> Tensor

    Computes :attr:`x` ≥ :attr:`y` element-wise.

    :attr:`y` should be a tensor whose shape is broadcastable with tensor :attr:`x`.
    
    Args:
        x (Tensor): the tensor to compare.
        y (Tensor): the tensor to compare.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> flow.F.broadcast_greater_equal(flow.tensor([0, 1, 2]), flow.tensor([1]))
        tensor([0, 1, 1], dtype=oneflow.int8)

""",
)

add_docstr(
    oneflow.F.broadcast_matmul,
    r"""
    broadcast_matmul(a: Tensor, b: Tensor, *, transpose_a=False: Bool, transpose_b=False: Bool, alpha=1.0: Double) -> Tensor 

    Performs a matrix-matrix product of matrices stored in :attr:`a` and :attr:`b`.

    :attr:`b` should be a tensor whose shape is broadcastable with tensor :attr:`a`.
    
    Args:
        a (Tensor): the first batch of matrices to be multiplied.
        b (Tensor): the matrice to be multiplied.
        transpose_a (Bool): whether to transpose the last two dimensions on :attr:`a`.
        transpose_b (Bool): whether to transpose the last two dimensions on :attr:`b`.
        alpha (Double): the scaling factor on matrix-matrix product.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.random.randn(10, 3, 4), dtype=flow.float32)
        >>> y = flow.Tensor(np.random.randn(4, 5), dtype=flow.float32)
        >>> res = flow.F.broadcast_matmul(x, y)
        >>> res.shape
        flow.Size([10, 3, 5])

""",
)

add_docstr(
    oneflow.F.broadcast_mul,
    r"""
    broadcast_mul(a: Tensor, b: Tensor, *, transpose_a=False: Bool, transpose_b=False: Bool, alpha=1.0: Double) -> Tensor 

    Computes the multiplication of :attr:`a` by :attr:`b` for each element.

    :attr:`b` should be a tensor whose shape is broadcastable with tensor :attr:`a`.
    
    Args:
        a (Tensor): the input tensor.
        b (Tensor):  the tensor to be multiplied to each element of :attr:`a`.
        transpose_a (Bool): whether to transpose the last two dimensions on :attr:`a`.
        transpose_b (Bool): whether to transpose the last two dimensions on :attr:`b`.
        alpha (Double): the scaling factor on multiplication product.
    
    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.random.randn(1, 1))
        >>> y = flow.Tensor(np.random.randn(2, 3))
        >>> out = flow.F.broadcast_mul(x, y)
        >>> out.shape
        flow.Size([2, 3])

""",
)
