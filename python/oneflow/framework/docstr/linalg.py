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
