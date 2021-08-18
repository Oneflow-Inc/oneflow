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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


class Sort(Module):
    def __init__(self, dim: int = -1, descending: bool = False) -> None:
        super().__init__()
        self.dim = dim
        direction = "DESCENDING" if descending else "ASCENDING"
        self._argsort_op = (
            flow.builtin_op("arg_sort")
            .Input("in")
            .Output("out")
            .Attr("direction", direction)
            .Build()
        )

    def forward(self, input):
        num_dims = len(input.shape)
        dim = self.dim if self.dim >= 0 else self.dim + num_dims
        assert 0 <= dim < num_dims, "dim out of range"
        if dim == num_dims - 1:
            indices = self._argsort_op(input)[0]
            return (flow.experimental.gather(input, indices, dim), indices)
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_dims, dim)
            x = flow.F.transpose(input, perm=perm)
            indices = self._argsort_op(x)[0]
            indices = flow.F.transpose(indices, perm=get_inversed_perm(perm))
            return (flow.experimental.gather(input, indices, dim), indices)


@register_tensor_op("sort")
def sort_op(input, dim: int = -1, descending: bool = False):
    """Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Args:
        input (oneflow.compatible.single_client.Tensor): The input Tensor.
        dim (int, optional): dimension to be sorted. Defaults to the last dim (-1).
        descending (bool, optional): controls the sorting order (ascending or descending).

    Returns:
        Tuple(oneflow.compatible.single_client.Tensor, oneflow.compatible.single_client.Tensor(dtype=int32)): A tuple of (values, indices), where
        where the values are the sorted values and the indices are the indices of the elements
        in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

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
    return Sort(dim=dim, descending=descending)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
