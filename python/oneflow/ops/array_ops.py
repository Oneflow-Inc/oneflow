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


def check_slice_tup_list(slice_tup_list, shape):
    ndim = len(shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndim:
        raise ValueError(
            "slice_tup_list must be a list or tuple with length less than or equal to number of dimensions of input tensor"
        )
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )
    start_list = []
    stop_list = []
    step_list = []
    for (slice_tup, dim_size) in zip(slice_tup_list, shape):
        if not isinstance(slice_tup, (tuple, list)) or len(slice_tup) != 3:
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )
        if not all((isinstance(idx, int) or idx is None for idx in slice_tup)):
            raise ValueError("element of slice tuple must int or None")
        (start, stop, step) = slice_tup
        if step is None:
            step = 1
        if step == 0:
            raise ValueError("slice step can't be 0")
        if start is None:
            start = 0 if step > 0 else np.iinfo(np.int64).max
        elif start < -dim_size or start >= dim_size:
            start, stop, step = 0, 0, 1
        if stop is None:
            stop = np.iinfo(np.int64).max if step > 0 else np.iinfo(np.int64).min
        elif stop < -dim_size - 1 or stop > dim_size:
            raise ValueError("slice start must be in range [-size-1, size]")
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
    return (start_list, stop_list, step_list)


def GetSliceAttrs(slice_tup_list, input_shape):
    ndim = len(input_shape)
    if not (isinstance(slice_tup_list, (list, tuple)) and len(slice_tup_list) <= ndim):
        raise ValueError(
            "slice_tup_list must be a list or tuple with length less than or equal to number of dimensions of input tensor"
        )
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )
    start_list = []
    stop_list = []
    step_list = []
    for (slice_tup, dim_size) in zip(slice_tup_list, input_shape):
        if not (isinstance(slice_tup, (tuple, list)) and len(slice_tup) == 3):
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )
        if not all((isinstance(idx, int) or idx is None for idx in slice_tup)):
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
    return (start_list, stop_list, step_list)
