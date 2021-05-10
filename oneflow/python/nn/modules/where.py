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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op


class Where(Module):
    def __init__(self) -> None:
        super().__init__()

        self._where_op = (
            flow.builtin_op("where")
            .Input("condition")
            .Input("x")
            .Input("y")
            .Output("out")
            .Build()
        )

    def forward(self, x, y, condition):
        return self._where_op(x, y, condition)[0]


@oneflow_export("tmp.where")
@register_tensor_op("where")
def where_op(condition, x, y):
    return Where()(condition=condition, x=x, y=y)
