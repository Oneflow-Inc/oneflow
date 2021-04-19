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
import numpy as np
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module


@oneflow_export("MaskedFill")
@register_op_by_module("masked_fill")
@register_tensor_op_by_module("masked_fill")
class MaskedFill(Module):
    r"""
    Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is True. 
    The shape of :attr:`mask` must be broadcastable with the shape of the underlying tensor.

    Args:
        mask (BoolTensor) – the boolean mask
        value (float) – the value to fill in with

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

    """

    def __init__(self) -> None:
        super().__init__()
        self._where_op = flow.builtin_op("where").Input("condition").Input("x").Input("y").Output("out").Build()

    def forward(self, input, mask, value):
        in_shape = tuple(input.shape)
        value_like_x = flow.Tensor(*in_shape)
        value_like_x.fill_(value)
        return self._where_op(mask, value_like_x, input)[0]


