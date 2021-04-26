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
from typing import Optional, Sequence


@oneflow_export("Transpose")
@register_op_by_module("tmp.transpose")
class Transpose(Module):
    r"""This operator transposes the specified axis of input Tensor.

    Args:
        a (oneflow.Tensor): The input tensor.
        perm (Sequence[int], optional): The list of dimension permutation. Defaults to None.
        conjugate (bool, optional): Still Unavailable. Defaults to False.
        batch_axis_non_change (bool, optional): deprecated. Defaults to False.

    Raises:
        NotImplementedError: The attribute `conjugate` still unavailable.

    Returns:
        oneflow.Tensor: A transposed tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np


        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        out = flow.tmp.transpose(input, perm=(0, 2, 3, 1))

        # out.shape (2, 5, 3, 6)

    """

    def __init__(
        self,
        perm: Sequence[int] = None,
        conjugate: bool = False,
        batch_axis_non_change: bool = False,
    ) -> None:
        super().__init__()

        assert isinstance(perm, (tuple, list))

        if conjugate:
            raise NotImplementedError

        self._op = (
            flow.builtin_op("transpose")
            .Input("input")
            .Output("output")
            .Attr("perm", perm)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@register_tensor_op_by_module("tmp.transpose")
def transpose_op(tensor, /, perm=None, conjugate=False, batch_axis_non_change=False):
    return Transpose(perm, conjugate, batch_axis_non_change)(tensor)
