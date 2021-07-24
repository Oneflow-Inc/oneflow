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
from typing import Optional

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    if dim_num == 1:
        return (False, None)
    if axis < 0:
        axis += dim_num
    assert axis >= 0
    assert axis < dim_num
    need_transpose = False
    permute = list(range(dim_num))
    if axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return (need_transpose, permute)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
