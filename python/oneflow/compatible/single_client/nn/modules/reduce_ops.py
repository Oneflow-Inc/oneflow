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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
