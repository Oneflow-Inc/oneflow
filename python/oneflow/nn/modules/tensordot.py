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
import oneflow
from typing import Union, List, Tuple


def tensordot(
    a, b, dims: Union[oneflow._oneflow_internal.Tensor, int, List[List[int]], Tuple[List[int]]] = 2
):
    if not isinstance(dims, (oneflow._oneflow_internal.Tensor, int, list, tuple)):
        raise TypeError(
            f"oneflow.tensordot expects dims to be one of oneflow.Tensor, int, Tuple[List[int], List[int]] or List[List[int]] containing two lists, but got {type(dims)}"
        )

    if isinstance(dims, int):
        return oneflow._C.tensordot(a, b, dims)
    elif isinstance(dims, (list, tuple)):
        assert len(dims) == 2
        dim_a = list(dims[0])
        dim_b = list(dims[1])
    elif isinstance(dims, oneflow._oneflow_internal.Tensor):
        assert len(dims) == 2
        dim_a = dims[0].tolist()
        dim_b = dims[1].tolist()

    return oneflow._C.tensordot(a, b, dim_a, dim_b)
