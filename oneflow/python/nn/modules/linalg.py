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

from typing import Optional, Sequence, Sized, Union
import collections
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util
from oneflow.python.framework.tensor import register_tensor_op_by_module


class MatMul(Module):
    def __init__(
        self,
        transpose_a: bool = False,
        transpose_b: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._op_matmul = (
            flow.builtin_op("matmul", name)
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Build()
        )

        self._op_batch_matmul = (
            flow.builtin_op("batch_matmul", name)
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", transpose_a)
            .Attr("transpose_b", transpose_b)
            .Build()
        )

    def forward(self, a, b):
        assert len(a.shape) == len(b.shape)
        assert len(a.shape) >= 2
        if len(a.shape) == 2:
            return self._op_matmul(a, b)[0]
        else:
            return self._op_batch_matmul(a, b)[0]


def matmul(
    a,
    b,
    transpose_a: bool = False,
    transpose_b: bool = False,
    name: Optional[str] = None,
):
    return MatMul(transpose_a, transpose_b, name)(a, b)


if __name__ == "__main__":
    import numpy as np

    flow.enable_eager_execution(True)
    a = flow.Tensor(np.random.randn(2, 3))
    b = flow.Tensor(np.random.randn(3, 2))
    c = matmul(a, b)
    print(c.shape)

    a = flow.Tensor(np.random.randn(5, 2, 3))
    b = flow.Tensor(np.random.randn(5, 3, 2))
    c = matmul(a, b)
    print(c.shape)
