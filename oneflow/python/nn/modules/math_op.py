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


@oneflow_export("To")
class To(Module):
    r"""

    """

    def __init__(self, dtype: flow.dtype, name: Optional[str] = None):

        if name is None:
            name = id_util.UniqueStr("To_")
        self.dtype = dtype
        self._op = (
            flow.builtin_op("cast")
            .Name(name)
            .Input("in")
            .Output("out")
            .Attr("dtype", dtype)
            .Build()
        )

    # return (
    #     flow.user_op_builder(name)
    #     .Op("cast")
    #     .Input("in", [x])
    #     .Output("out")
    #     .Attr("dtype", dtype)
    #     .Build()
    #     .InferAndTryRun()
    #     .RemoteBlobList()[0]
    # )

    def forward(self, x):
        if x.dtype == self.dtype:
            return x
        else:
            return self._op(x)[0]


import numpy as np

if __name__ == "__main__":
    flow.enable_eager_execution()
    x = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
    print(x.dtype)

    to = flow.To(flow.float32)
    # y = to(x)
    # print(y.dtype)
