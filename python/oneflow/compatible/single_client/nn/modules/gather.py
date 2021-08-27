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
from typing import List, Optional, Tuple

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import Tensor, register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Gather(Module):
    def __init__(self, dim: int = 0, sparse_grad: bool = False):
        super().__init__()
        assert sparse_grad is False, "Only support bool = False for now!"
        self.dim = dim

    def forward(self, input, index):
        assert self.dim < len(
            index.shape
        ), "Value of dim is out of range(dim should be less than len(index.shape))"
        assert len(input.shape) == len(
            index.shape
        ), "Dimensions of input and index should equal"
        for i in range(0, len(input.shape)):
            if self.dim == i:
                continue
            else:
                assert (
                    input.shape[i] == index.shape[i]
                ), "Dimensions of input and index should be same except at dim"
        return flow.F.dim_gather(input, index, dim=self.dim)


@register_tensor_op("gather")
def gather_op(input, index, dim=0, sparse_grad=False):
    """Gathers values along an axis specified by `dim`.

    For a 3-D tensor the output is specified by:

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :attr:`input` and :attr:`index` must have the same number of dimensions.
    It is also required that ``index.size(d) <= input.size(d)`` for all
    dimensions ``d != dim``.  :attr:`out` will have the same shape as :attr:`index`.
    Note that ``input`` and ``index`` do not broadcast against each other.

    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to gather

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = np.random.randn(3, 4, 3, 5)
        >>> index = np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5))
        >>> output = flow.gather(flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=1)
        >>> output.shape
        flow.Size([3, 4, 3, 5])

    """
    return Gather(dim, sparse_grad)(input, index)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
