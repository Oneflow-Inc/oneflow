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

import oneflow as flow
from oneflow.ops.array_ops import parse_slice_tuple_list


def slice_op(input, slice_tup_list: Sequence[Tuple[int, int, int]]):
    """Extracts a slice from a tensor.
    The `slice_tup_list` assigns the slice indices in each dimension, the format is (start, stop, step).
    The operator will slice the tensor according to the `slice_tup_list`.

    Args:
        input: A `Tensor`.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.Tensor(np.random.randn(3, 6, 9).astype(np.float32))
        >>> tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
        >>> y = flow.slice(input, slice_tup_list=tup_list)
        >>> y.shape
        oneflow.Size([3, 3, 2])
    """
    (start, stop, step) = parse_slice_tuple_list(slice_tup_list, input.shape)
    return flow._C.slice(input, start, stop, step)


def slice_update_op(input, update, slice_tup_list: Sequence[Tuple[int, int, int]]):
    """Update a slice of tensor `x`. Like `x[start:stop:step] = update`.

    Args:
        x: A `Tensor`, whose slice will be updated.
        update: A `Tensor`, indicate the update content.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.Tensor(np.array([1, 1, 1, 1, 1]).astype(np.float32))
        >>> update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
        tensor([1., 2., 3., 4., 1.], dtype=oneflow.float32)

    """

    (start, stop, step) = parse_slice_tuple_list(slice_tup_list, input.shape)
    if update.dtype != input.dtype:
        update = update.to(dtype=input.dtype)
    return flow._C.slice_update(input, update, start, stop, step, inplace=True)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
