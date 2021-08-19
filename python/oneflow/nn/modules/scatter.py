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
from oneflow.framework.tensor import Tensor
from oneflow.nn.module import Module


__all__ = ["scatter", "scatter_add"]


def scatter(input, dim, index, src):
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

    Returns:
        Tensor: The scatterd Tensor. 

    For example: 

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.ones((3,5))*2
        >>> index = flow.tensor(np.array([[0,1,2],[0,1,4]], ), dtype=flow.int32)
        >>> src = flow.Tensor(np.array([[0,10,20,30,40],[50,60,70,80,90]]))
        >>> out = flow.scatter(input, 1, index, src)
        >>> out
        tensor([[ 0., 10., 20.,  2.,  2.],
                [50., 60.,  2.,  2., 70.],
                [ 2.,  2.,  2.,  2.,  2.]], dtype=oneflow.float32)

    """

    return flow.F.scatter(input, dim, index, src)


def scatter_add(input, dim, index, src):
    r"""This operator scatter the src with addition operation according to index along dim into the input.

    Take a 3-D blob as example, the output is specified by:
    
    .. code-block:: python

        input[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
        input[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
        input[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

    Args:
        input (Tensor): The input blob.
        dim (int): The axis along which to index
        index (Tensor): The index blob of elements to scatter. 
        src (Tensor): The source blob whose elements will be scatterd and added to output.

    Returns:
        Tensor: The scatterd Tensor. 

    For example: 

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.ones((3,5))*2
        >>> index = flow.tensor(np.array([[0,1,2],[0,1,4]], ), dtype=flow.int32)
        >>> src = flow.Tensor(np.array([[0,10,20,30,40],[50,60,70,80,90]]))
        >>> out = flow.scatter_add(input, 1, index, src)
        >>> out
        tensor([[ 2., 12., 22.,  2.,  2.],
                [52., 62.,  2.,  2., 72.],
                [ 2.,  2.,  2.,  2.,  2.]], dtype=oneflow.float32)

    """

    assert type(src) in [
        flow.Tensor
    ], f"type of src must be oneflow.Tensor, but %s givien" % type(src)

    return flow.F.dim_scatter_add(input, index, src, dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
