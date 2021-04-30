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
import oneflow.python.framework.id_util as id_util
from typing import Optional, Sequence


class Squeeze(Module):
    def __init__(self, axis: Optional[Sequence[int]] = None) -> None:
        super().__init__()

        self._op = (
            flow.builtin_op("squeeze")
            .Input("in")
            .Output("out")
            .Attr("axes", axis)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("tmp.squeeze")
@register_tensor_op("squeeze")
def squeeze_op(input, axis: Optional[Sequence[int]] = None):
    """This operator removes the specified dimention which size is 1 of the input Tensor.
    If the `axis` is not specified, this operator will remove all the dimention which size is 1 of the input Tensor.

    The amount of element in return value is the same as Tensor `input`.

    Args:
        input (oneflow.Tensor): The input Tensor.
        axis (Optional[Sequence[int]], optional): The axis. Defaults to None.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    Example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        input = flow.Tensor(np.array([[[[1, 1, 1]]]]).astype(np.int32))
        out = flow.tmp.squeeze(input, axis=[1, 2]).numpy().shape

        # out.shape (1, 3)

    """
    return Squeeze(axis=axis)(input)
