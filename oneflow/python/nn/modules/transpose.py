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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional, Sequence


class Transpose(Module):
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

        if batch_axis_non_change:
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


@oneflow_export("transpose")
@register_tensor_op("transpose")
@experimental_api
def transpose_op(tensor, perm: Sequence[int] = None):
    r"""This operator transposes the specified axis of input Tensor.
    Args:
        tensor (oneflow.Tensor): The input tensor.
        perm (Sequence[int], optional): The list of dimension permutation. Defaults to None.
    Returns:
        oneflow.Tensor: A transposed tensor.
    For example:
    .. code-block:: python
        import oneflow.experimental as flow
        import numpy as np

        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        out = flow.transpose(input, perm=(0, 2, 3, 1))

        # out.shape (2, 5, 3, 6)
    """
    return Transpose(perm=perm)(tensor)
