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

from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module

from typing import Optional, List, Tuple


class Scatter_nd(Module):
    def __init__(self) -> None:
        super().__init__()
        self.scatter_nd_op = (
            flow.builtin_op("scatter_nd").Input("indices").Input("updates").Output("out").Build()
        )

    def forward(self, indices, updates):

        return self.scatter_nd_op(input,indices)[0]


@oneflow_export("scatter_nd")
@experimental_api
def scatter_nd__op(indices, updates):
    r"""Gathers values along an axis specified by `dim`.

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

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = np.random.randn(3, 4, 3, 5)
        >>> index = np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5))
        >>> output = flow.gather(flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=1)
        >>> output.shape
        flow.Size([3, 4, 3, 5])

    """
    return Scatter_nd()(indices, updates)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
