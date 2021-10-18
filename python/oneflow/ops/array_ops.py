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


def parse_slice_tuple_list(slice_tup_list, shape):
    ndim = len(shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndim:
        raise ValueError(
            "slice_tup_list must be a list or tuple with length less than or equal "
            "to number of dimensions of input tensor"
        )

    if len(slice_tup_list) < ndim:
        supple_ndim = ndim - len(slice_tup_list)
        slice_tup_list += type(slice_tup_list)([(None, None, None)] * supple_ndim)

    start_list, stop_list, step_list = [], [], []
    for (slice_tup, dim) in zip(slice_tup_list, shape):
        if not isinstance(slice_tup, (tuple, list)) or len(slice_tup) != 3:
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )

        if not all((isinstance(elem, int) or elem is None for elem in slice_tup)):
            raise ValueError("element of slice tuple must int or None")

        (start, stop, step) = slice_tup

        if step is None:
            step = 1

        if step == 0:
            raise ValueError("slice step can't be 0")

        if start is None:
            start = 0 if step > 0 else dim

        if stop is None:
            stop = dim if step > 0 else -dim - 1

        # start range is [-dim, dim-1]
        start = max(min(start, dim - 1), -dim)
        # stop range is [-dim-1, dim]
        stop = max(min(stop, dim), -dim - 1)

        reg_start = start if start >= 0 else start + dim
        reg_stop = stop if stop >= 0 else stop + dim

        if step > 0 and reg_stop < reg_start:
            stop = start

        if step < 0 and reg_start < reg_stop:
            stop = start

        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)

    return start_list, stop_list, step_list
