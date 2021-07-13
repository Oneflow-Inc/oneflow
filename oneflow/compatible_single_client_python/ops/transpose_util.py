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
from __future__ import absolute_import

from typing import Sequence


def is_perm(perm: Sequence[int],) -> bool:
    return list(range(len(perm))) == sorted(list(perm))


# get the perm when you want to transpose specified axis to the last dimension
def get_perm_when_transpose_axis_to_last_dim(num_axes: int, axis: int,) -> tuple:
    axis = axis if axis >= 0 else axis + num_axes
    assert 0 <= axis < num_axes, "axis out of range"
    perm = [dim if dim < axis else dim + 1 for dim in range(num_axes - 1)]
    perm.append(axis)
    return tuple(perm)


# x == transpose(transpose(x, perm), get_inversed_perm(perm))
def get_inversed_perm(perm: Sequence[int],) -> tuple:
    assert is_perm(perm)
    inversed_perm = [-1] * len(perm)
    for i in range(len(perm)):
        inversed_perm[perm[i]] = i
    return tuple(inversed_perm)
