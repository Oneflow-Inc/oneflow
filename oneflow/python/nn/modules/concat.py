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
from oneflow.python.framework.tensor import Tensor
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module
from typing import Optional, Sequence


@oneflow_export("Cat")
@register_op_by_module("cat")
class Cat(Module):
    """
    """

    def __init__(self, axis=0, max_dim_size: Optional[int] = None, n=2) -> None:
        super().__init__()
        self._op = flow.builtin_op("concat").Input("in", n).Output("out").Build()
        self.axis = axis
        self.max_dim_size = max_dim_size

    def forward(self, inputs):

        if len(inputs) == 1:
            return inputs[0]

        axis = self.axis
        max_dim_size = self.max_dim_size
        assert len(inputs) >= 2
        if axis < 0:
            axis += len(inputs[0].shape)
        assert axis >= 0 and axis < len(
            inputs[0].shape
        ), "axis must be in range [0, num_axes of inputs)"

        first_input_shape = inputs[0].shape
        dynamic_dim_size = 0
        for input in inputs:
            assert len(input.shape) == len(first_input_shape)
            for i in range(len(input.shape)):
                if i == axis:
                    dynamic_dim_size += input.shape[i]
                else:
                    assert input.shape[i] == first_input_shape[i]

        if max_dim_size is None:
            max_dim_size = dynamic_dim_size
        else:
            assert (
                max_dim_size >= dynamic_dim_size
            ), "max diemension size {} is too small to hold concatenated static dimension size {} along the given axis".format(
                max_dim_size, dynamic_dim_size
            )

        return self._op(*inputs, axis=axis, max_dim_size=max_dim_size)[0]
