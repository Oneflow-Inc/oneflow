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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import _check_axis


def _build_reduce_op(op_type_name, keepdims):
    return (
        flow.builtin_op(op_type_name)
        .Input("input_tensor")
        .Output("output_tensor")
        .Attr("keepdims", keepdims)
        .Build()
    )


class Sum(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self._op = _build_reduce_op("reduce_sum", keepdims)

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@register_tensor_op("sum")
def _sum(input, dim=None, keepdim=False):
    """Computes the sum of row of elements in a tensor in the given axis, if the axis is None, sum of all elements will be caculated.
    
    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.sum(input)
        tensor([21.], dtype=oneflow.float32)
        >>> flow.sum(input, dim=0)
        tensor([5., 7., 9.], dtype=oneflow.float32)
        >>> flow.sum(input, dim=1)
        tensor([ 6., 15.], dtype=oneflow.float32)

    """
    return Sum(dim, keepdim)(input)


class Mean(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        if axis is None:
            self.axes = []
        else:
            self.axes = list(axis) if isinstance(axis, collections.Sized) else [axis]

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        reduce_sum = flow.experimental.sum(input, dim=self.axis, keepdim=self.keepdims)
        reduce_count = 1
        if len(self.axes) == 0:
            for dim in input.shape:
                reduce_count *= dim
        else:
            for i in self.axes:
                reduce_count *= input.shape[i]
        return flow.experimental.mul(reduce_sum, 1.0 / reduce_count)


@register_tensor_op("mean")
def _mean(input, dim=None, keepdim=False):
    """Computes the mean of row of elements in a tensor in the given axis, if the axis is None, mean of all elements will be caculated.
    
    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.mean(input)
        tensor([3.5], dtype=oneflow.float32)
        >>> flow.mean(input, dim=0)
        tensor([2.5, 3.5, 4.5], dtype=oneflow.float32)
        >>> flow.mean(input, dim=1)
        tensor([2., 5.], dtype=oneflow.float32)

    """
    return Mean(dim, keepdim)(input)


class Min(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self._op = _build_reduce_op("reduce_min", keepdims)

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@register_tensor_op("min")
def _min(input, dim=None, keepdim=False):
    """Computes the minimum value of all elements in the input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.min(input)
        tensor([1.], dtype=oneflow.float32)
        >>> flow.min(input, dim=0)
        tensor([2., 1., 3.], dtype=oneflow.float32)
        >>> flow.min(input, dim=1)
        tensor([1., 2.], dtype=oneflow.float32)

    """
    return Min(dim, keepdim)(input)


class Max(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self._op = _build_reduce_op("reduce_max", keepdims)

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@register_tensor_op("max")
def _max(input, dim=None, keepdim=False):
    """Computes the maximum value of all elements in the input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.max(input)
        tensor([6.], dtype=oneflow.float32)
        >>> flow.max(input, dim=0)
        tensor([4., 6., 5.], dtype=oneflow.float32)
        >>> flow.max(input, dim=1)
        tensor([5., 6.], dtype=oneflow.float32)

    """
    return Max(dim, keepdim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
