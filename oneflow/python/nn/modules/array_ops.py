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
import oneflow as flow

from typing import Optional, Sequence, Sized, Union
import collections
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util
from oneflow.python.framework.tensor import register_tensor_op_by_module


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


class SliceV2(Module):
    r"""
    """

    def __init__(
        self,
        slice_tup_list: Sequence[Tuple[int, int, int]],
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)
        self._op = (
            flow.builtin_op("slice", name)
            .Input("x")
            .Output("y")
            .Attr("start", start)
            .Attr("stop", stop)
            .Attr("step", step)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


def slice(
    x: flow.Tensor,
    begin: Sequence[int],
    size: Sequence[int],
    name: Optional[str] = None,
):

    ndim = len(x.shape)
    if not isinstance(begin, (list, tuple)) or len(begin) != ndim:
        raise ValueError(
            "begin must be a list/tuple with the same length as input tensor's number of dimensions"
        )

    if not all(isinstance(b, int) or b is None for b in begin):
        raise ValueError("element of begin must be a int or None")

    if not isinstance(size, (list, tuple)) or len(size) != ndim:
        raise ValueError(
            "size must be a list/tuple with the same length as input tensor's number of dimensions."
        )

    if not all(isinstance(s, int) or s is None for s in size):
        raise ValueError("element of size must be a int or None")

    slice_tup_list = []
    for b, s, dim_size in zip(begin, size, x.shape):
        start, stop, step = (None, None, 1)
        if b is not None:
            if b < -dim_size or b >= dim_size:
                raise ValueError("element of begin is out of range")
            start = b

        if s is not None:
            if s == -1:
                stop = dim_size
            else:
                if s <= 0 or s > dim_size:
                    raise ValueError("element of size is invalid")
                if b + s < dim_size:
                    stop = b + s

        slice_tup_list.append((start, stop, step))

    return SliceV2(slice_tup_list, name=name)(x)


if __name__ == "__main__":
    import numpy as np

    flow.enable_eager_execution(True)
    x = flow.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    y = slice(x, begin=[None, 0], size=[None, 2])
    print(y.numpy())
