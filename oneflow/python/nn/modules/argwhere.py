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
from oneflow.python.framework.dtype import dtypes
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional


class Argwhere(Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        if dtype == None:
            dtype = flow.int32
        self._op = (
            flow.builtin_op("argwhere")
            .Input("input")
            .Output("output")
            .Output("output_size")
            .Attr("dtype", dtype)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("argwhere")
@register_tensor_op("argwhere")
@experimental_api
def argwhere_op(x, dtype: Optional[flow.dtype] = None):
    """

    """
    return Argwhere(dtype=dtype)(x)
