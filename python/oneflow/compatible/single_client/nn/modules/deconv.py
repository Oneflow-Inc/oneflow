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
import math

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn import init
from oneflow.compatible.single_client.python.nn.common_types import _size_2_t
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.modules.utils import _pair


def slice(x, begin, size):
    ndim = len(x.shape)
    if not isinstance(begin, (list, tuple)) or len(begin) != ndim:
        raise ValueError(
            "begin must be a list/tuple with the same length as input tensor's number of dimensions"
        )
    if not all((isinstance(b, int) or b is None for b in begin)):
        raise ValueError("element of begin must be a int or None")
    if not isinstance(size, (list, tuple)) or len(size) != ndim:
        raise ValueError(
            "size must be a list/tuple with the same length as input tensor's number of dimensions."
        )
    if not all((isinstance(s, int) or s is None for s in size)):
        raise ValueError("element of size must be a int or None")
    slice_tup_list = []
    for (b, s, dim_size) in zip(begin, size, x.shape):
        (start, stop, step) = (None, None, 1)
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
    return flow.experimental.slice(x, slice_tup_list)


class ConvUtil(object):
    @classmethod
    def split(cls, x, axis, split_num):
        split_len = x.shape[axis] // split_num
        result_list = []
        slice_begin = [0] * len(x.shape)
        slice_size = [-1] * len(x.shape)
        slice_size[axis] = split_len
        for i in range(split_num):
            slice_begin[axis] = i * split_len
            result = slice(x, slice_begin, slice_size)
            result_list.append(result)
        return result_list


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
