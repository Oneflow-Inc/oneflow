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
from typing import Optional


@oneflow_export("Expand_Dims")
@register_op_by_module("tmp.expand_dims")
class Expand_Dims(Module):
    """This operator inserts a dimention at the specified axis in the input Tensor.
    The size of new dimension can only be 1, and the amount of element in return value is the same as Tensor `input`.

    Args:
        input (oneflow.Tensor): The input Tensor.
        axis (int): The specified dimension index.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        import numpy as np
        import oneflow as flow

        input = flow.Tensor(np.random.randn(2, 6, 5), dtype=flow.float32)
        out = flow.tmp.expand_dims(input, axis=-1).numpy().shape

        # out (2, 6, 5, 1)

    """

    def __init__(self, axis: int) -> None:
        super().__init__()

        self._op = (
            flow.builtin_op("expand_dims")
            .Input("in")
            .Output("out")
            .Attr("axis", axis)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]
