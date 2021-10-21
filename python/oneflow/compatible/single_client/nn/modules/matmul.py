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
from typing import Optional, Sequence

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class MatMul(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return flow.F.matmul(a, b)


@register_tensor_op("matmul")
def matmul_op(a, b):
    """This operator applies matrix multiplication to two Tensor.

    Args:
        a (oneflow.compatible.single_client.Tensor): A Tensor
        b (oneflow.compatible.single_client.Tensor): A Tensor

    Returns:
        oneflow.compatible.single_client.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> input1 = flow.Tensor(np.random.randn(2, 6), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.random.randn(6, 5), dtype=flow.float32)
        >>> of_out = flow.matmul(input1, input2)
        >>> of_out.shape
        flow.Size([2, 5])

    """
    return flow.F.matmul(a, b)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
