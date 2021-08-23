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
from typing import Sequence, Tuple

import numpy as np

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.ops.array_ops import (
    GetSliceAttrs,
    check_slice_tup_list,
)


class Slice(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x):
        return flow.F.slice(x, start=self.start, stop=self.stop, step=self.step)


def slice_op(x, slice_tup_list: Sequence[Tuple[int, int, int]]):
    """Extracts a slice from a tensor.
    The `slice_tup_list` assigns the slice indices in each dimension, the format is (start, stop, step).
    The operator will slice the tensor according to the `slice_tup_list`.

    Args:
        x: A `Tensor`.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.random.randn(3, 6, 9).astype(np.float32))
        >>> tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
        >>> y = flow.slice(input, slice_tup_list=tup_list)
        >>> y.shape
        flow.Size([3, 3, 2])
    """
    (start, stop, step) = check_slice_tup_list(slice_tup_list, x.shape)
    return Slice(start, stop, step)(x)


class SliceUpdate(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x, update):
        return flow.F.slice_update(
            x, update, start=self.start, stop=self.stop, step=self.step
        )


def slice_update_op(x, update, slice_tup_list: Sequence[Tuple[int, int, int]]):
    """Update a slice of tensor `x`. Like `x[start:stop:step] = update`. 

    Args:
        x: A `Tensor`, whose slice will be updated.
        update: A `Tensor`, indicate the update content.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.array([1, 1, 1, 1, 1]).astype(np.float32))
        >>> update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> y = flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
        >>> y.numpy()
        array([1., 2., 3., 4., 1.], dtype=float32)
    """
    (start, stop, step) = GetSliceAttrs(slice_tup_list, x.shape)
    return SliceUpdate(start, stop, step)(x, update)


class LogicalSliceAssign(Module):
    def __init__(
        self, start: Tuple[int, ...], stop: Tuple[int, ...], step: Tuple[int, ...]
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x, update):
        if update.dtype != x.dtype:
            update = update.to(dtype=x.dtype)
        return flow.F.logical_slice_assign(
            x, update, start=self.start, stop=self.stop, step=self.step
        )


def logical_slice_assign_op(x, update, slice_tup_list: Sequence[Tuple[int, int, int]]):
    """Update a slice of tensor `x`(in-place). Like `x[start:stop:step] = update`. 

    Args:
        x: A `Tensor`, whose slice will be updated.
        update: A `Tensor`, indicate the update content.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.array([1, 1, 1, 1, 1]).astype(np.float32))
        >>> update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> y = flow.tmp.logical_slice_assign(input, update, slice_tup_list=[[1, 4, 1]])
    """
    "[summary]\n\n    Returns:\n        [type]: [description]\n    "
    (start, stop, step) = GetSliceAttrs(slice_tup_list, x.shape)
    return LogicalSliceAssign(start, stop, step)(x, update)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
