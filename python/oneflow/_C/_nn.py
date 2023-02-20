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
import builtins
import warnings
from oneflow.framework.tensor import Tensor
from typing import overload, Tuple, Any

_device = flow.device
_bool = builtins.bool
_dtype = flow.dtype


@overload
def _parse_to(
    device: _device,
    dtype: _dtype,
    non_blocking: _bool,
    copy: _bool,
    *,
    memory_format: Any,
) -> Tuple[_device, _dtype, _bool, Any]:
    ...


@overload
def _parse_to(
    dtype: _dtype, non_blocking: _bool, copy: _bool, *, memory_format: Any
) -> Tuple[_device, _dtype, _bool, Any]:
    ...


@overload
def _parse_to(
    tensor: Tensor, non_blocking: _bool, copy: _bool, *, memory_format: Any
) -> Tuple[_device, _dtype, _bool, Any]:
    ...


def _parse_to(*args, **kwargs):
    new_args = list()
    # If device is single int, replace it with flow.device("cuda:{device}")
    if len(args) > 0 and isinstance(args[0], int):
        new_args.append(flow.randn((3, 4)))
        new_args.append(flow.device(f"cuda:{args[0]}"))
    else:
        new_args = list(args)
        new_args.insert(0, flow.randn((3, 4)))

    for i in range(1, len(new_args)):
        if not isinstance(new_args[i], Tensor):
            if new_args[i] == int or new_args[i] == float:
                # dtype support python int or float
                new_args = flow.int64 if new_args[i] == int else flow.float64

    if "memory_format" in kwargs:
        warnings.warn("oneflow temporarily support contiguous format.")

    # check whether non_blocking and copy are both passed
    flag = False
    index = -1
    for i, data in enumerate(new_args):
        if isinstance(data, bool):
            if flag:
                index = i - 1  # the index of non_blocking
            else:
                flag = True

    if flag and index != -1:
        non_blocking_param = new_args.pop(index)
        if non_blocking_param:
            raise ValueError("oneflow temporarily supports 'non_blocking is False' ")
        result = flow._C.to(*new_args)
    else:
        result = flow._C.to(*new_args)
    # this return (device, dtype, non_blocking, memory_format)
    # non_blocking only support False, use None to represent memory_format
    return (result.device, result.dtype, False, None)
