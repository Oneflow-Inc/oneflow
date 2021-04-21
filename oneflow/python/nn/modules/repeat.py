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
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module
from oneflow.python.ops.transpose_util import (
    get_perm_when_transpose_axis_to_last_dim,
    get_inversed_perm,
)


@oneflow_export("Repeat")
@register_tensor_op_by_module("tmp.repeat")
@register_op_by_module("tmp.repeat")
class Repeat(Module):
    r"""
    """

    def __init__(self, repeat_size) -> None:
        super().__init__()
        self.repeat_size = repeat_size

    def forward(self, input):
        repeat_size = self.repeat_size
        input_shape = input.shape
        new_input_shape = []
        for i in range(max(len(input_shape), len(repeat_size))):
            if i >= len(input_shape):
                x = 1
            else:
                x = input_shape[i]
            if i >= len(repeat_size):
                y = 1
            else:
                y = repeat_size[i]
            new_input_shape.append(x * y)
        print(new_input_shape)
        return flow.tmp.expand(input, expand_size=new_input_shape)
