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
import numpy as np
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.ops.array_ops import _check_slice_tup_list, _GetSliceAttrs
from typing import Sequence, Tuple


class Slice(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("slice")
            .Input("x")
            .Output("y")
            .Attr("start", start)
            .Attr("stop", stop)
            .Attr("step", step)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("slice")
@experimental_api
def slice_op(x, slice_tup_list: Sequence[Tuple[int, int, int]]):
    r"""Extracts a slice from a tensor.
    The `slice_tup_list` assigns the slice indices in each dimension, the format is (start, stop, step).
    The operator will slice the tensor according to the `slice_top_list`.

    Args:
        x: A `Tensor`.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example: 

    .. code-block:: python 

        import oneflow as flow 
        import oneflow.typing as tp 

        input = flow.Tensor(np.random.randn(3, 6, 9).astype(np.float32))
        tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
        y = flow.experimental.slice(input, slice_tup_list=tup_list)

        # y.shape >> flow.Size([3, 3, 2]
    """
    start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)
    return Slice(start, stop, step)(x)


class SliceUpdate(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("slice_update")
            .Input("x")
            .Input("update")
            .Output("y")
            .Attr("start", start)
            .Attr("stop", stop)
            .Attr("step", step)
            .Build()
        )

    def forward(self, x, update):
        return self._op(x, update)[0]


@oneflow_export("slice_update")
@experimental_api
def slice_update_op(x, update, slice_tup_list: Sequence[Tuple[int, int, int]]):
    r"""Update a slice of tensor `x`. Like `x[start:stop:step] = update`. 

    Args:
        x: A `Tensor`, whose slice will be updated.
        update: A `Tensor`, indicate the update content.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example: 

    .. code-block:: python 

        import oneflow as flow 
        import oneflow.typing as tp

        input = flow.Tensor(np.array([1, 1, 1, 1, 1]).astype(np.float32))
        update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        y = flow.experimental.slice_update(input, update, slice_tup_list=[[1, 4, 1]])

        # [1. 2. 3. 4. 1.] 
    """
    start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)
    return SliceUpdate(start, stop, step)(x, update)


class LogicalSliceAssign(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("logical_slice_assign")
            .Input("ref")
            .Input("value")
            .Attr("start", start)
            .Attr("stop", stop)
            .Attr("step", step)
            .Build()
        )

    def forward(self, x, update):
        return self._op(x, update)


# NOTE: conflict with exist userop: flow.experimental.logical_slice_assign, so use tmp.logical_slice_assign
@oneflow_export("tmp.logical_slice_assign")
@experimental_api
def logical_slice_assign_op(x, update, slice_tup_list: Sequence[Tuple[int, int, int]]):
    r"""Update a slice of tensor `x`(in-place). Like `x[start:stop:step] = update`. 

    Args:
        x: A `Tensor`, whose slice will be updated.
        update: A `Tensor`, indicate the update content.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example: 

    .. code-block:: python 

        import oneflow as flow 
        import oneflow.typing as tp

        input = flow.Tensor(np.array([1, 1, 1, 1, 1]).astype(np.float32))
        update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        y = flow.tmp.logical_slice_assign(input, update, slice_tup_list=[[1, 4, 1]])

        # [1. 2. 3. 4. 1.] 
    """
    start, stop, step = _GetSliceAttrs(slice_tup_list, x.shape)
    return LogicalSliceAssign(start, stop, step)(x, update)
