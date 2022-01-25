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


def min_op(input, dim=None, keepdim=False):
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


def sum_op(input, dim=None, keepdim=False):
    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_sum(input, axis=axis_checked, keepdims=keepdim)


def mean_op(input, dim=None, keepdim=False):
    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_mean(input, axis=axis_checked, keepdims=keepdim)


def prod_op(input, dim=None, keepdim=False):
    axis_checked = _check_axis(dim, input.shape)
    if len(axis_checked) == 0:
        return input
    return flow._C.reduce_prod(input, axis_checked, keepdim)


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
