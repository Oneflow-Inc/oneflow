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

import collections.abc as container_abcs
from itertools import repeat
from typing import List

import oneflow as flow


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def _getint():
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return int(x[0])
        return int(x)

    return parse


_getint = _getint()
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _handle_size_arg(size):
    if len(size) == 0:
        return size
    assert len(size) > 0, "size of tensor doesn't exists"
    if isinstance(size[0], (list, tuple, flow.Size)):
        assert (
            len(size) == 1
        ), "shape should be specified by tuple of int size, not tuple of list"
        size = size[0]
    return size


def _reverse_repeat_tuple(t, n):
    """Reverse the order of `t` and repeat each element for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple((x for x in reversed(t) for _ in range(n)))


def _list_with_default(out_size, defaults):
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            "Input dimension should be at least {}".format(len(out_size) + 1)
        )
    return [
        v if v is not None else d
        for (v, d) in zip(out_size, defaults[-len(out_size) :])
    ]


def _check_axis(axis, shape):
    ndim = len(shape)
    if axis is None:
        axis = list(range(len(shape)))
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(axis, (list, tuple)), "Invalid axis {}".format(axis)
    axis = list(axis)
    for i in range(len(axis)):
        assert (
            -ndim <= axis[i] <= ndim - 1
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -ndim, ndim - 1, axis[i]
        )
        if axis[i] < 0:
            axis[i] = axis[i] + ndim
    return axis


def _generate_output_size(input_size, output_size):
    new_output_size = []
    assert len(input_size) - 2 == len(
        output_size
    ), f"the length of 'output_size' does not match the input size, {len(input_size) - 2} expected"
    for i in range(len(output_size)):
        if output_size[i] is None:
            new_output_size.append(input_size[i + 2])
        else:
            assert isinstance(
                output_size[i], int
            ), "numbers in 'output_size' should be integer"
            new_output_size.append(output_size[i])
    return tuple(new_output_size)
