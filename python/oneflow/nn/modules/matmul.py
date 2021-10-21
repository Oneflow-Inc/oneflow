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

import oneflow as flow
import oneflow.framework.id_util as id_util
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


@register_tensor_op("matmul")
def matmul_op(input, other):
    """This operator applies matrix multiplication to two Tensor.

    Args:
        a (oneflow.Tensor): A Tensor
        b (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input1 = flow.tensor(np.random.randn(2, 6), dtype=flow.float32)
        >>> input2 = flow.tensor(np.random.randn(6, 5), dtype=flow.float32)
        >>> of_out = flow.matmul(input1, input2)
        >>> of_out.shape
        oneflow.Size([2, 5])

    """
    return flow._C.matmul(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
