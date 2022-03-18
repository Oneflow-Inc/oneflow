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
    oneflow.max,
    """max(input, dim, keepdim)
    Computes the maximum value of all elements in the input tensor.
    
    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    Returns:
        Tensor or Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int64)): If :attr:`dim` is `None`, returns 
        the maximum value of all elements in the `input` tensor. Otherwise, returns a tuple of Tensor (values, indices), 
        where the `values` are the maximum value of all elements in the `input` tensor,
        the `indices` are the indices of the elements in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.max(input)
        tensor(6., dtype=oneflow.float32)
        >>> (values, indices) = flow.max(input, dim=1)
        >>> values
        tensor([5., 6.], dtype=oneflow.float32)
        >>> indices
        tensor([2, 1], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.min,
    """min(input, dim, keepdim)
    Computes the minimum value of all elements in the input tensor.
    
    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    Returns:
        Tensor or Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int64)): If :attr:`dim` is `None`, returns 
        the minimum value of all elements in the `input` tensor. Otherwise, returns a tuple of Tensor (values, indices), 
        where the `values` are the minimum value of all elements in the `input` tensor,
        the `indices` are the indices of the elements in the original input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.min(input)
        tensor(1., dtype=oneflow.float32)
        >>> (values, indices) = flow.min(input, dim=1)
        >>> values
        tensor([1., 2.], dtype=oneflow.float32)
        >>> indices
        tensor([1, 0], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.sum,
    """Computes the sum of row of elements in a tensor in the given axis, if the axis is None, sum of all elements will be caculated.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.sum(input)
        tensor(21., dtype=oneflow.float32)
        >>> flow.sum(input, dim=0)
        tensor([5., 7., 9.], dtype=oneflow.float32)
        >>> flow.sum(input, dim=1)
        tensor([ 6., 15.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.mean,
    """Computes the mean of row of elements in a tensor in the given axis, if the axis is None, mean of all elements will be caculated.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.mean(input)
        tensor(3.5000, dtype=oneflow.float32)
        >>> flow.mean(input, dim=0)
        tensor([2.5000, 3.5000, 4.5000], dtype=oneflow.float32)
        >>> flow.mean(input, dim=1)
        tensor([2., 5.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.prod,
    r"""Computes the product of row of elements in a tensor in the given axis.
    
    note: `if the dim is None, it will return a tensor with only one element whose value is the product of all elements of input.`

    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to prod

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.prod(input)
        tensor(720., dtype=oneflow.float32)
        >>> flow.prod(input, dim=0)
        tensor([ 4., 10., 18.], dtype=oneflow.float32)
        >>> flow.prod(input, dim=1)
        tensor([  6., 120.], dtype=oneflow.float32)

    """,
)
