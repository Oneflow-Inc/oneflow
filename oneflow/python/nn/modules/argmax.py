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


@oneflow_export("Argmax")
@register_tensor_op_by_module("argmax")
@register_op_by_module("argmax")
class Argmax(Module):
    r"""
    Returns the largest value of the :attr:`input` at specified axis.
    Args:
        {input}
    Keyword args:
        {axis}
    """

    def __init__(self, axis=-1) -> None:
        super().__init__()
        self._op_softmax_last_dim = (
            flow.builtin_op("argmax").Input("in").Output("out").Build()
        )
        self._op_transpose_1 = (
            flow.builtin_op("transpose").Input("input").Output("output").Build()
        )
        self._op_transpose_2 = (
            flow.builtin_op("transpose").Input("input").Output("output").Build()
        )
        self._op_expand = (
            flow.builtin_op("expand_dims").Input("in").Output("out").Build()
        )
        self._op_squeeze = flow.builtin_op("squeeze").Input("in").Output("out").Build()
        self.axis = axis

    def forward(self, input):
        num_axes = len(input.shape)
        self.axis = self.axis if self.axis >= 0 else self.axis + num_axes
        assert 0 <= self.axis < num_axes, "axis out of range"
        if self.axis == num_axes - 1:
            return self._op_softmax_last_dim(input)[0]
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, self.axis)
            x = self._op_transpose_1(input, perm=perm)[0]
            x = self._op_softmax_last_dim(x)[0]
            x = self._op_expand(x, axis=-1)[0]
            x = self._op_transpose_2(x, perm=get_inversed_perm(perm))[0]
            x = self._op_squeeze(x, axes=[self.axis])[0]
            return x
