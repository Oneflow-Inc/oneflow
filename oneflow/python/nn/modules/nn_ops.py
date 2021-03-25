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


class BiasAdd(Module):
    def __init__(
        self, data_format: Optional[str] = None, name: Optional[str] = None,
    ) -> None:
        super().__init__()

        if data_format is None:
            bias_add_axis = 1
        else:
            if data_format.startswith("NC"):
                bias_add_axis = 1
            elif data_format.startswith("N") and data_format.endswith("C"):
                bias_add_axis = len(value.shape) - 1
            else:
                raise ValueError("data_format must be of the form `N...C` or `NC...`")

        self._op = (
            flow.builtin_op("bias_add", name)
            .Input("a")  # value
            .Input("b")  # bias
            .Output("out")
            .Attr("axis", bias_add_axis)
            .Build()
        )

    def forward(self, value, bias):
        return self._op(value, bias)[0]


def nn_bias_add(a, b, data_format=None, name=None):
    return BiasAdd(data_format)(a, b)


if __name__ == "__main__":
    import numpy as np

    flow.enable_eager_execution(True)
    a = flow.Tensor(np.random.randn(1, 64, 128, 128))
    b = flow.Tensor(np.random.randn(64,))
    out = nn_bias_add(a, b)
    print(out.numpy())
