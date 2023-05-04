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
    oneflow.linalg.cross,
    """linalg.cross(input, other, dim=-1) -> Tensor

    Computes the cross product of two 3-dimensional vectors.

    Supports input of float and double dtypes. 
    Also supports batches of vectors, for which it computes the product along the dimension dim. 
    In this case, the output has the same batch dimensions as the inputs broadcast to a common shape.

    The documentation is referenced from: https://pytorch.org/docs/1.11/generated/torch.linalg.cross.html

    Args:
        input (Tensor): the first input tensor.
        other (Tensor): the second input tensor.
        dim (int, optional): the dimension along which to take the cross-product. Default: `-1`

    Raises:
        RuntimeError:  If after broadcasting ``input.size(dim) != 3`` or ``other.size(dim) != 3``.
    
    Examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.tensor([[ -0.3956, 1.1455,  1.6895],
        ...                  [ -0.5849, 1.3672,  0.3599],
        ...                  [ -1.1626, 0.7180, -0.0521],
        ...                  [ -0.1339, 0.9902, -2.0225]])
        >>> b = flow.tensor([[ -0.0257, -1.4725, -1.2251],
        ...                  [ -1.1479, -0.7005, -1.9757],
        ...                  [ -1.3904,  0.3726, -1.1836],
        ...                  [ -0.9688, -0.7153,  0.2159]])
        >>> flow.linalg.cross(a, b)
        tensor([[ 1.0844, -0.5281,  0.6120],
                [-2.4491, -1.5687,  1.9791],
                [-0.8304, -1.3036,  0.5651],
                [-1.2329,  1.9883,  1.0551]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.cross,
    """cross(input, other, dim=None) -> Tensor

    Returns the cross product of vectors in dimension `dim` of `input` and `other`.

    Supports input of float and double dtypes. 
    Also supports batches of vectors, for which it computes the product along the dimension `dim`. 
    In this case, the output has the same batch dimensions as the inputs.

    If `dim` is not given, it defaults to the first dimension found with the size 3. Note that this might be unexpected.

    The documentation is referenced from: https://pytorch.org/docs/1.11/generated/torch.cross.html

    .. warning::
        This function may change in a future PyTorch release to match the default behaviour in :func:`oneflow.linalg.cross`. We recommend using :func:`oneflow.linalg.cross`.

    Args:
        input (Tensor): the first input tensor.
        other (Tensor): the second input tensor.
        dim (int, optional): the dimension to take the cross-product in. Default: `None`
    
    Examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.tensor([[ -0.3956, 1.1455,  1.6895],
        ...                  [ -0.5849, 1.3672,  0.3599],
        ...                  [ -1.1626, 0.7180, -0.0521],
        ...                  [ -0.1339, 0.9902, -2.0225]])
        >>> b = flow.tensor([[ -0.0257, -1.4725, -1.2251],
        ...                  [ -1.1479, -0.7005, -1.9757],
        ...                  [ -1.3904,  0.3726, -1.1836],
        ...                  [ -0.9688, -0.7153,  0.2159]])
        >>> flow.cross(a, b)
        tensor([[ 1.0844, -0.5281,  0.6120],
                [-2.4491, -1.5687,  1.9791],
                [-0.8304, -1.3036,  0.5651],
                [-1.2329,  1.9883,  1.0551]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.linalg.multi_dot,
    """multi_dot(input, other, dim=None) -> Tensor

    Efficiently multiplies two or more matrices by reordering the multiplications so that the fewest arithmetic operations are performed.
    Supports inputs of float, double, cfloat and cdouble dtypes. This function does not support batched inputs.
    Every tensor in tensors must be 2D, except for the first and last which may be 1D. If the first tensor is a 1D vector of shape (n,) it is treated as a row vector of shape (1, n), similarly if the last tensor is a 1D vector of shape (n,) it is treated as a column vector of shape (n, 1).
    If the first and last tensors are matrices, the output will be a matrix. However, if either is a 1D vector, then the output will be a 1D vector.
    Differences with numpy.linalg.multi_dot:

    - Unlike numpy.linalg.multi_dot, the first and last tensors must either be 1D or 2D whereas NumPy allows them to be nD.

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.linalg.multi_dot.html

    .. warning::
        This function does not broadcast.

    .. note::
        This function is implemented by chaining :func:`oneflow.mm()` calls after computing the optimal matrix multiplication order.

    Args:
        tensors (Sequence[Tensor]): two or more tensors to multiply. The first and last tensors may be 1D or 2D. Every other tensor must be 2D.

    Examples:

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> from oneflow.linalg import multi_dot

        >>> multi_dot([flow.tensor([1, 2]), flow.tensor([2, 3])])
        tensor(8, dtype=oneflow.int64)
        >>> multi_dot([flow.tensor([[1, 2]]), flow.tensor([2, 3])])
        tensor([8], dtype=oneflow.int64)
        >>> multi_dot([flow.tensor([[1, 2]]), flow.tensor([[2], [3]])])
        tensor([[8]], dtype=oneflow.int64)

        >>> A = flow.arange(2 * 3).view(2, 3)
        >>> B = flow.arange(3 * 2).view(3, 2)
        >>> C = flow.arange(2 * 2).view(2, 2)
        >>> multi_dot((A, B, C))
        tensor([[ 26,  49],
                [ 80, 148]], dtype=oneflow.int64)
    """,
)
