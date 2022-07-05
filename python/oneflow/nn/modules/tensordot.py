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
import warnings


def tensordot(
    a,
    b,
    dims: Union[oneflow.Tensor, int, List[List[int]], Tuple[List[int]]] = 2,
    out=None,
):
    if out is not None:
        raise NotImplementedError(
            "tensordot with `out` parameter which is not None is not yet implemented"
        )
    if not isinstance(dims, (oneflow.Tensor, int, list, tuple)):
        raise TypeError(
            f"oneflow.tensordot expects dims to be one of oneflow.Tensor, int, Tuple[List[int], List[int]] or List[List[int]] containing two lists, but got {type(dims)}"
        )

    if isinstance(dims, int):
        return oneflow._C.tensordot(a, b, dims)
    elif isinstance(dims, (list, tuple)):
        assert (
            len(dims) == 2
        ), f"The list/tuple of dims must contain two lists, got {len(dims)}"
        dim_a = list(dims[0])
        dim_b = list(dims[1])
    elif isinstance(dims, oneflow.Tensor):
        warnings.warn(
            "tensordot doesn't support nn.Graph when the type of `dims` is oneflow.Tensor, because it needs synchronization."
        )
        if dims.numel() == 1:
            return oneflow._C.tensordot(a, b, dims.item())
        assert (
            dims.dim() == 2
        ), f"The dims tensor must have two dimensions, got {dims.dim()}"
        assert (
            len(dims) == 2 and dims.dim() == 2
        ), f"The dims tensor must have two rows, got {len(dims)}"
        dim_a = dims[0].tolist()
        dim_b = dims[1].tolist()

    return oneflow._C.tensordot(a, b, dim_a, dim_b)
