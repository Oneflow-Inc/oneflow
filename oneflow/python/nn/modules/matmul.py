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
import oneflow.python.framework.id_util as id_util
from typing import Optional, Sequence


class MatMul(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        assert len(a.shape) >= 2, "Tensor a's dim should >=2"
        assert len(b.shape) >= 2, "Tensor b's dim should >=2"

        if len(a.shape) == len(b.shape):
            if len(a.shape) == 2:
                res = flow.F.matmul(a, b)
            else:
                res = flow.F.batch_matmul(a, b)
        else:
            # NOTE: support broadcast b to a only for now
            assert (
                len(b.shape) == 2
            ), "Not support number of dimensions of a being less than number of dimensions of b!"
            res = flow.F.broadcast_matmul(a, b)

        return res


@oneflow_export("matmul")
@register_tensor_op("matmul")
@experimental_api
def matmul_op(a, b):
    r"""This operator applies matrix multiplication to two Tensor.

    Args:
        a (oneflow.Tensor): A Tensor
        b (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example: 

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> input1 = flow.Tensor(np.random.randn(2, 6), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.random.randn(6, 5), dtype=flow.float32)
        >>> of_out = flow.matmul(input1, input2)
        >>> print(of_out.shape)
        flow.Size([2, 5])
    """
    return MatMul()(a, b)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
