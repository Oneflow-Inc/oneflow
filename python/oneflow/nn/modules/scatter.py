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


class Scatter(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, dim, index, src, reduce):
        assert type(src) in [
            flow.Tensor,
            float,
        ], f"type of src must be oneflow.Tensor or float, but %s givien" % type(src)

        assert reduce in [
            "add",
            "multiply",
            None,
        ], "reduce must be 'add', 'multiply' or None"

        if isinstance(src, flow.Tensor):
            if reduce == "add":
                return flow.F.dim_scatter_add(input, index, src, dim)
            elif reduce == "multiply":
                return flow.F.dim_scatter_mul(input, index, src, dim)
            return flow.F.dim_scatter(input, index, src, dim)
        elif isinstance(src, float):
            if reduce == "add":
                return flow.F.dim_scatter_add_scalar(input, index, src, dim)
            elif reduce == "multiply":
                return flow.F.dim_scatter_mul_scalar(input, index, src, dim)
            return flow.F.dim_scatter_scalar(input, index, src, dim)


@oneflow_export("scatter")
@experimental_api
def scatter_op(input, dim, index, src, reduce: Optional[str] = None):
    r"""This operator writes the elements specified by `index` along with the axis 
    `dim` from the `src` into the `input`.

    Take a 3-D blob as example, the output is specified by:
    
    .. code-block:: python

        input[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        input[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        input[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    input, index and src (if it is a Tensor) should all have the same number of dimensions. 
    It is also required that index.shape(d) <= src.shape(d) for all dimensions d, 
    and that index.shape(d) <= self.shape(d) for all dimensions d != dim.
    Note that index and src do not broadcast.

    Args:
        input (Tensor): The input blob.
        dim (int): The axis along which to index
        index (Tensor): The index blob of elements to scatter. 
        src (Tensor or float): The source blob whose elements will be scatterd and updated to output.
        reduce (string): reduction operation to apply, can be either 'add' or 'multiply'.

    Returns:
        Tensor: The scatterd Tensor. 

    For example: 

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np

        >>> input = flow.ones((3,5))*2
        >>> index = flow.tensor(np.array([[0,1,2],[0,1,4]], ), dtype=flow.int32)
        >>> src = flow.Tensor(np.array([[0,10,20,30,40],[50,60,70,80,90]]))
        >>> out = flow.scatter(input, 1, index, src)
        >>> out
        tensor([[ 0., 10., 20.,  2.,  2.],
                [50., 60.,  2.,  2., 70.],
                [ 2.,  2.,  2.,  2.,  2.]], dtype=oneflow.float32)
        >>> out = flow.scatter(input, 1, index, src, reduce="add")
        >>> out
        tensor([[ 2., 12., 22.,  2.,  2.],
                [52., 62.,  2.,  2., 72.],
                [ 2.,  2.,  2.,  2.,  2.]], dtype=oneflow.float32)
        >>> out = flow.scatter(input, 1, index, src, reduce="multiply")
        >>> out
        tensor([[  0.,  20.,  40.,   2.,   2.],
                [100., 120.,   2.,   2., 140.],
                [  2.,   2.,   2.,   2.,   2.]], dtype=oneflow.float32)
        >>> out = flow.scatter(input, 1, index, 3.14)
        >>> out
        tensor([[3.14, 3.14, 3.14, 2.  , 2.  ],
                [3.14, 3.14, 2.  , 2.  , 3.14],
                [2.  , 2.  , 2.  , 2.  , 2.  ]], dtype=oneflow.float32)
        >>> out = flow.scatter(input, 1, index, 3.14, reduce="add")
        >>> out
        tensor([[5.14, 5.14, 5.14, 2.  , 2.  ],
                [5.14, 5.14, 2.  , 2.  , 5.14],
                [2.  , 2.  , 2.  , 2.  , 2.  ]], dtype=oneflow.float32)
        >>> out = flow.scatter(input, 1, index, 3.14, reduce="multiply")
        >>> out
        tensor([[6.28, 6.28, 6.28, 2.  , 2.  ],
                [6.28, 6.28, 2.  , 2.  , 6.28],
                [2.  , 2.  , 2.  , 2.  , 2.  ]], dtype=oneflow.float32)

    """

    return Scatter()(input, dim, index, src, reduce)


@register_tensor_op
@experimental_api
def scatter_tensor_op(input, dim, index, src, reduce: Optional[str] = None):
    r"""
    In-place version of :func:`oneflow.experimental.scatter`

    """

    return Scatter()(input, dim, index, src, reduce)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
