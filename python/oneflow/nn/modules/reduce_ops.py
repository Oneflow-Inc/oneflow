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
import collections
from typing import Optional, Sequence, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_axis


def max_op(input, dim=None, keepdim=False):
    """Computes the maximum value of all elements in the input tensor.
    
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

    """

    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    if dim == None:
        return flow._C.reduce_max(input, axis=axis_checked, keepdims=keepdim)
    else:
        return (
            flow._C.reduce_max(input, axis=axis_checked, keepdims=keepdim),
            input.argmax(dim=dim, keepdim=keepdim),
        )


@register_tensor_op("max")
def max_tensor_op(input, dim=None, keepdim=False):
    """
    input.max(dim, index) -> Tensor
    See :func:`oneflow.max`
    """

    return max_op(input, dim, keepdim)


def min_op(input, dim=None, keepdim=False):
    """Computes the minimum value of all elements in the input tensor.
    
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

    """

    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    if dim == None:
        return flow._C.reduce_min(input, axis=axis_checked, keepdims=keepdim)
    else:
        return (
            flow._C.reduce_min(input, axis=axis_checked, keepdims=keepdim),
            input.argmin(dim=dim, keepdim=keepdim),
        )


@register_tensor_op("min")
def min_tensor_op(input, dim=None, keepdim=False):
    """
    input.min(dim, index) -> Tensor
    See :func:`oneflow.min`
    """

    return min_op(input, dim, keepdim)


def sum_op(input, dim=None, keepdim=False):
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

    """

    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_sum(input, axis=axis_checked, keepdims=keepdim)


@register_tensor_op("sum")
def sum_tensor_op(input, dim=None, keepdim=False):
    """
    input.sum(dim, index) -> Tensor
    See :func:`oneflow.sum`
    """

    return sum_op(input, dim, keepdim)


def mean_op(input, dim=None, keepdim=False):
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

    """
    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_mean(input, axis=axis_checked, keepdims=keepdim)


@register_tensor_op("mean")
def mean_tensor_op(input, dim=None, keepdim=False):
    """
    input.mean(dim, index) -> Tensor
    See :func:`oneflow.mean`
    """

    return mean_op(input, dim, keepdim)


def prod_op(input, dim=None, keepdim=False):
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

    """
    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_prod(input, axis_checked, keepdim)


@register_tensor_op("prod")
def prod_tensor_op(input, dim=None, keepdim=False):
    """
    input.prod(dim, index) -> Tensor
    See :func:`oneflow.prod`
    """

    return prod_op(input, dim, keepdim)


def all_op(input, dim=None, keepdim=False):
    """Computes if all elements in the input tensor to true.
    
    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    Returns:
        Tensor(oneflow.Tensor(dtype=int8)): If :attr:`dim` is `None`, returns 
        the logical all value of all elements in the `input` tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.arange(0, 3)
        >>> flow.all(input)
        tensor(False, dtype=oneflow.bool)

    """

    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_all(input, axis=axis_checked, keepdims=keepdim)


def any_op(input, dim=None, keepdim=False):
    """Computes if any elements in the input tensor to true.
    
    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    Returns:
        Tensor(oneflow.Tensor(dtype=int8)): If :attr:`dim` is `None`, returns 
        the logical any value of all elements in the `input` tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.arange(0, 3)
        >>> flow.any(input)
        tensor(True, dtype=oneflow.bool)

    """

    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_any(input, axis=axis_checked, keepdims=keepdim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
