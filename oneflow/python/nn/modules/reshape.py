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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Sequence
from functools import reduce
import operator


def infer_shape(x, shape):
    dim_index_need_infer = shape.index(-1) if shape.count(-1) == 1 else None
    in_elem_cnt = reduce(operator.mul, x.shape, 1)
    out_elem_cnt = reduce(operator.mul, shape, 1)
    if dim_index_need_infer is not None:
        assert (in_elem_cnt % out_elem_cnt) == 0
        shape[dim_index_need_infer] = int(abs(in_elem_cnt / out_elem_cnt))
    else:
        assert in_elem_cnt == out_elem_cnt
    return shape


class Reshape(Module):
    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()

        assert isinstance(shape, tuple) or isinstance(shape, list)
        shape = list(shape)
        assert all(dim == -1 or dim > 0 for dim in shape)
        assert shape.count(-1) <= 1

        self._op = (
            flow.builtin_op("reshape")
            .Input("in")
            .Output("out")
            .Attr("shape", shape)
            .Build()
        )
        self.shape = shape

    def forward(self, x):
        new_shape = infer_shape(x, self.shape)
        return self._op(x, shape=new_shape)[0]


@oneflow_export("reshape")
@register_tensor_op("reshape")
@experimental_api
def reshape_op(x, shape: Sequence[int] = None):
    """This operator reshapes a Tensor.

    We can set one dimension in `shape` as `-1`, the operator will infer the complete shape.

    Args:
        x: A Tensor.
        shape: Shape of the output tensor.
    Returns:
        A Tensor has the same type as `x`.

    For example:

    .. code-block:: python

        import numpy as np
        import oneflow.experimental as flow

        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        ).astype(np.float32)
        input = flow.Tensor(x)

        y = flow.reshape(input, shape=[2, 2, 2, -1]).numpy().shape

        # y (2, 2, 2, 2)

    """
    return Reshape(shape=shape)(x)
