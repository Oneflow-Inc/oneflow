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
    # TODO: implement _parse_to natively
    result = flow.tensor([]).to(*args, **kwargs)

    return (result.device, result.dtype, False, None)
