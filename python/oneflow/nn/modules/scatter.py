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
from oneflow.nn.modules.module import Module

__all__ = ["scatter", "scatter_add", "scatter_nd", "tensor_scatter_nd_update"]


def scatter(input, dim, index, src, *, reduce=None):
    r"""This operator writes the elements specified by `index` along with the axis 
    `dim` from the `src` into the `input`.

    Take a 3-D blob as example, the output is specified by:
    
    .. code-block:: python

        input[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        input[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        input[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    input, index and src (if it is a Tensor) should all have the same number of dimensions. 
    It is also required that index.shape(d) <= src.shape(d) for all dimensions d, 
    and that index.shape(d) <= input.shape(d) for all dimensions d != dim.
    Note that index and src do not broadcast.

    .. warning::
        When indices are not unique, the behavior is non-deterministic (one of the values from src will be picked arbitrarily) 
        and the gradient will be incorrect (it will be propagated to all locations in the source that correspond to the same index)!
    
    .. note::
        The backward pass is implemented only for ``src.shape == index.shape``.
    
    Additionally accepts an optional ``reduce`` argument that allows specification of an optional reduction operation, 
    which is applied to all values in the tensor ``src`` into ``input`` at the indicies specified in the ``index``. 
    For each value in ``src``, the reduction operation is applied to an index in ``input`` which is specified by its index in ``src`` for ``dimension != dim`` 
    and by the corresponding value in ``index`` for ``dimension = dim``.

    Given a 3-D tensor and reduction using the multiplication operation, input is updated as:

    .. code-block:: python

        input[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
        input[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
        input[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

    Reducing with the addition operation is the same as using :func:`oneflow.scatter_add()`.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.Tensor.scatter\_.html.

    Args:
        input (Tensor): The input blob.
        dim (int): The axis along which to index
        index (Tensor): The index blob of elements to scatter. 
        src (Tensor or float): The source blob whose elements will be scatterd and updated to output.
        reduce (str, optional): Reduction operation to apply, can be either ``add`` or ``multiply``.

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
    return flow._C.scatter(input, dim, index, src, reduce=reduce)


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

    return flow._C.scatter_add(input, dim, index, src)


def scatter_nd(index, update, shape):
    """This operator inserts the elements in `update` according to the `index` and create a new Tensor.

    Args:
        index: The indices of `update`. Its type should be `flow.int`.
        update: The update Tensor.
        shape (Sequence[int]): The constant tensor shape, the constant tensor elements are all zero.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> index = flow.tensor(np.array([[1], [6], [4]]), dtype=flow.int)
        >>> update = flow.tensor(np.array([10.2, 5.1, 12.7]), dtype=flow.float)
        >>> out = flow.scatter_nd(index, update, [8])
        >>> out
        tensor([ 0.0000, 10.2000,  0.0000,  0.0000, 12.7000,  0.0000,  5.1000,  0.0000],
               dtype=oneflow.float32)

    """
    return flow._C.scatternd(index, update, shape)


def tensor_scatter_nd_update(tensor, indices, updates):
    r"""
    This operation creates a new tensor by applying sparse updates to the input tensor.
    This is similar to an index assignment.

    This operator is very similar to :meth:`scatter_nd`, except that the updates are scattered onto an existing
    tensor (as opposed to a zero-tensor).

    Args:
        tensor: The tensor will be scattered.
        indices: The indices of ``update``. Its type should be `flow.int`.
        update: The update Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> tensor = flow.arange(8)
        >>> indices = flow.tensor([[1], [3], [5]])
        >>> updates = flow.tensor([-1, -2, -3])
        >>> flow.tensor_scatter_nd_update(tensor, indices, updates)
        tensor([ 0, -1,  2, -2,  4, -3,  6,  7], dtype=oneflow.int64)

    """
    return flow._C.tensor_scatter_nd_update(tensor, indices, updates)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
