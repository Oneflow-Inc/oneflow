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

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple


@oneflow_export("nn.Linear")
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.use_bias = bias
        self.weight = flow.nn.Parameter(
            flow.Tensor(
                out_features,
                in_features,
                data_initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
            )
        )

        if bias:

            self.bias = flow.nn.Parameter(
                flow.Tensor(
                    out_features,
                    data_initializer=flow.random_uniform_initializer(
                        minval=-1, maxval=1
                    ),
                )
            )

            self._bias_add_op = (
                flow.builtin_op("bias_add")
                .Name("bias_add")
                .Input("a")
                .Input("b")
                .Output("out")
                .Attr("axis", 1)
                .Build()
            )

        self._op = (
            flow.builtin_op("matmul")
            .Name("matmul")
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", False)
            .Attr("transpose_b", True)
            .Build()
        )

    def reset_parameters(self) -> None:
        raise NotImplementedError()

    def forward(self, x):
        if self.use_bias:
            res = self._bias_add_op(self._op(x, self.weight)[0], self.bias)[0]
        else:
            res = self._op(x, self.weight)[0]
        return res
