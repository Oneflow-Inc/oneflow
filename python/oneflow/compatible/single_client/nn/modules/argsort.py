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


class Argsort(Module):
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
            return self._argsort_op(input)[0]
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_dims, dim)
            x = flow.F.transpose(input, perm=perm)
            x = self._argsort_op(x)[0]
            return flow.F.transpose(x, perm=get_inversed_perm(perm))


@register_tensor_op("argsort")
def argsort_op(input, dim: int = -1, descending: bool = False):
    """This operator sorts the input Tensor at specified dim and return the indices of the sorted Tensor.

    Args:
        input (oneflow.compatible.single_client.Tensor): The input Tensor.
        dim (int, optional): dimension to be sorted. Defaults to the last dim (-1).
        descending (bool, optional): controls the sorting order (ascending or descending).

    Returns:
        oneflow.compatible.single_client.Tensor: The indices of the sorted Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([[10, 2, 9, 3, 7],
        ...               [1, 9, 4, 3, 2]]).astype("float32")
        >>> input = flow.Tensor(x)
        >>> output = flow.argsort(input)
        >>> output
        tensor([[1, 3, 4, 2, 0],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, descending=True)
        >>> output
        tensor([[0, 2, 4, 3, 1],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, dim=0)
        >>> output
        tensor([[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]], dtype=oneflow.int32)

    """
    return Argsort(dim=dim, descending=descending)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
