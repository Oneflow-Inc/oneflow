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
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


@register_tensor_op("sort")
def sort_op(input, dim: int = -1, descending: bool = False):
    """Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Args:
        input (oneflow.Tensor): The input Tensor.
        dim (int, optional): dimension to be sorted. Defaults to the last dim (-1).
        descending (bool, optional): controls the sorting order (ascending or descending).

    Returns:
        Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int32)): A tuple of (values, indices), where
        where the values are the sorted values and the indices are the indices of the elements
        in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=np.float32)
        >>> input = flow.Tensor(x)
        >>> (values, indices) = flow.sort(input)
        >>> values
        tensor([[1., 2., 3., 7., 8.],
                [1., 2., 3., 4., 9.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4, 1, 3, 2],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> (values, indices) = flow.sort(input, descending=True)
        >>> values
        tensor([[8., 7., 3., 2., 1.],
                [9., 4., 3., 2., 1.]], dtype=oneflow.float32)
        >>> indices
        tensor([[2, 3, 1, 4, 0],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> (values, indices) = flow.sort(input, dim=0)
        >>> values
        tensor([[1., 3., 4., 3., 2.],
                [1., 9., 8., 7., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 0, 1, 1, 0],
                [1, 1, 0, 0, 1]], dtype=oneflow.int32)
 
    """
    num_dims = len(input.shape)
    dim = dim if dim >= 0 else dim + num_dims
    direction = "DESCENDING" if descending else "ASCENDING"
    assert 0 <= dim < num_dims, "dim out of range"
    if dim == num_dims - 1:
        indices = flow._C.arg_sort(input, direction)
        return (flow.gather(input, dim, indices), indices)
    else:
        perm = get_perm_when_transpose_axis_to_last_dim(num_dims, dim)
        x = flow._C.transpose(input, perm=perm)
        indices = flow._C.arg_sort(x, direction)
        indices = flow._C.transpose(indices, perm=get_inversed_perm(perm))
        return (flow.gather(input, dim, indices), indices)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
