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
from typing import Optional, Sequence, Sized, Union, List, Tuple
from oneflow.python.nn.modules.array_ops import transpose


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    assert dim_num >= 2
    if axis < 0:
        axis += dim_num
    assert axis >= 0
    assert axis < dim_num

    need_transpose = False
    permute = list(range(dim_num))
    if axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return need_transpose, permute


@oneflow_export("nn.Sigmoid")
@register_tensor_op_by_module("sigmoid")
@register_op_by_module("sigmoid")
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("sigmoid").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.ReLU")
@register_tensor_op_by_module("relu")
@register_op_by_module("relu")
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("relu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.Softmax")
class Softmax(Module):
    def __init__(
        self, axis: Optional[int] = None, name: Optional[str] = None,
    ):
        super().__init__()

        if axis is None:
            axis = -1
        self.axis = axis

        self._op = flow.builtin_op("softmax", name).Input("in").Output("out").Build()

    def forward(self, x):
        print(x.shape)
        need_transpose, permute = _softmax_need_transpose(x, self.axis)
        if need_transpose:
            logits = flow.transpose(logits, perm=permute)

        res = self._op(x)[0]
        if need_transpose:
            res = transpose(res, perm=permute)
        return res


@oneflow_export("nn.GeLU")
@register_tensor_op_by_module("gelu")
@register_op_by_module("gelu")
class GeLU(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("gelu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res

if __name__ == "__main__":
    flow.enable_eager_execution(True)
    import numpy as np

    x = flow.Tensor(np.array([[1, 2, 1, 5, 4]]))
    out = Softmax(1)(x)
    print(out.numpy())
