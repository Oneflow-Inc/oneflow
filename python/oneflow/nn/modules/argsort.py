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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module
from oneflow.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


def argsort_op(input, dim: int = -1, descending: bool = False):
    num_dims = len(input.shape)
    dim = dim if dim >= 0 else dim + num_dims
    direction = "DESCENDING" if descending else "ASCENDING"
    assert 0 <= dim < num_dims, "dim out of range"
    if dim == num_dims - 1:
        return flow._C.arg_sort(input, direction)
    else:
        perm = get_perm_when_transpose_axis_to_last_dim(num_dims, dim)
        x = flow._C.transpose(input, perm=perm)
        x = flow._C.arg_sort(x, direction)
        return flow._C.transpose(x, perm=get_inversed_perm(perm))


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
