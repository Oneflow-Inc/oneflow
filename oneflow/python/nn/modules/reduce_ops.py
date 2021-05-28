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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.modules.utils import _check_axis


def _build_math_binary_elementwise_op(math_op):
    return flow.builtin_op(math_op).Input("x").Input("y").Output("z").Build()


class Sum(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()

        self.axis = axis
        self.keepdims = keepdims
        self._op = (
            flow.builtin_op("reduce_sum")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", keepdims)
            .Build()
        )

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@oneflow_export("sum")
@register_tensor_op("sum")
@experimental_api
def _sum(input, dim=None, keepdims=False):
    r"""Computes the sum of row of elements in a tensor in the given axis, if the axis is None, sum of all elements will be caculated.
    
    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = flow.sum(input, dim=(2,1))

    """

    return Sum(dim, keepdims)(input)


class Min(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()

        self.axis = axis
        self.keepdims = keepdims
        self._op = (
            flow.builtin_op("reduce_min")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", keepdims)
            .Build()
        )

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@oneflow_export("min")
@register_tensor_op("min")
@experimental_api
def _min(input, dim=None, keepdims=False):
    r"""Computes the min of row of elements in a tensor in the given axis, if the axis is None, min of all elements will be caculated.
    
    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = flow.min(input, dim=(2,1))

    """

    return Min(dim, keepdims)(input)


class Max(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()

        self.axis = axis
        self.keepdims = keepdims
        self._op = (
            flow.builtin_op("reduce_max")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", keepdims)
            .Build()
        )

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@oneflow_export("max")
@register_tensor_op("max")
@experimental_api
def _max(input, dim=None, keepdims=False):
    r"""Computes the max of row of elements in a tensor in the given axis, if the axis is None, sum of all elements will be caculated.
    
    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = flow.max(input, dim=(2,1))

    """

    return Max(dim, keepdims)(input)
