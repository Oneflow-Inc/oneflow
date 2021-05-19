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
from typing import Optional, Sequence, Tuple


def _check_slice_tup_list(slice_tup_list, shape):
    ndim = len(shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndim:
        raise ValueError(
            "slice_tup_list must be a list or tuple with length "
            "less than or equal to number of dimensions of input tensor"
        )

    # if length of slice_tup_list is less than number of dimensions of x, fill it to length of ndims reduce 1
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )

    start_list = []
    stop_list = []
    step_list = []

    for slice_tup, dim_size in zip(slice_tup_list, shape):
        if not isinstance(slice_tup, (tuple, list)) or len(slice_tup) != 3:
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )

        if not all(isinstance(idx, int) or idx is None for idx in slice_tup):
            raise ValueError("element of slice tuple must int or None")

        (start, stop, step) = slice_tup
        if step is None:
            step = 1

        if step == 0:
            raise ValueError("slice step can't be 0")

        if start is None:
            start = 0 if step > 0 else np.iinfo(np.int64).max
        elif start < -dim_size or start >= dim_size:
            raise ValueError("slice start must be in range [-size, size)")

        if stop is None:
            stop = np.iinfo(np.int64).max if step > 0 else np.iinfo(np.int64).min
        elif stop < -dim_size - 1 or stop > dim_size:
            raise ValueError("slice start must be in range [-size-1, size]")

        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)

    return start_list, stop_list, step_list


# Get slice attrs for slice_assign and logical_slice
# Note the step in slice_tup_list must be greater than 0
# as slice_assign and logical_slice only support step > 0
def _GetSliceAttrs(slice_tup_list, input_shape):
    ndim = len(input_shape)
    if not (isinstance(slice_tup_list, (list, tuple)) and len(slice_tup_list) <= ndim):
        raise ValueError(
            "slice_tup_list must be a list or tuple with length "
            "less than or equal to number of dimensions of input tensor"
        )

    # Right extends slice_tup_list with [None, None, None] if len(slice_tup_list) < len(input_shape)
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )

    start_list = []
    stop_list = []
    step_list = []

    for slice_tup, dim_size in zip(slice_tup_list, input_shape):
        if not (isinstance(slice_tup, (tuple, list)) and len(slice_tup) == 3):
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )

        if not all(isinstance(idx, int) or idx is None for idx in slice_tup):
            raise ValueError("element of slice tuple must int or None")

        (start, stop, step) = slice_tup
        if step is None:
            step = 1

        if step <= 0:
            raise ValueError("slice_assign/logical_slice step must be greater than 0")

        if start is None:
            start = 0
        elif start < -dim_size or start >= dim_size:
            raise ValueError(
                "slice_assign/logical_slice start must be in range [-size, size)"
            )
        elif start < 0:
            start += dim_size

        if stop is None:
            stop = dim_size
        elif stop < -dim_size or stop > dim_size:
            raise ValueError(
                "slice_assign/logical_slice start must be in range [-size, size]"
            )
        elif stop < 0:
            stop += dim_size

        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)

    return start_list, stop_list, step_list


class Slice(Module):
    def __init__(self, start: int, stop: int, step: int) -> None:
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
        y = flow.tmp.slice(input, slice_tup_list=tup_list)

        # y.shape >> flow.Size([3, 3, 2]
    """
    start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)
    return Slice(start, stop, step)(x)


class SliceUpdate(Module):
    def __init__(self, start: int, stop: int, step: int) -> None:
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
        y = flow.tmp.slice_update(input, update, slice_tup_list=[[1, 4, 1]])

        # [1. 2. 3. 4. 1.] 
    """
    start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)
    return SliceUpdate(start, stop, step)(x, update)


class LogicalSliceAssign(Module):
    def __init__(self, start: int, stop: int, step: int) -> None:
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


@oneflow_export("tmp.logical_slice_assign")
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
