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
from oneflow.framework.tensor import register_tensor_op


@register_tensor_op("transpose")
def transpose_op(input, dim0, dim1):
    """Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

    The resulting out tensor shares its underlying storage with the input tensor, so changing the content of one would change the content of the other.

    Args:
        input (oneflow.Tensor): The input tensor.
        dim0 (int): the first dimension to be transposed.
        dim1 (int): the second dimension to be transposed.
    Returns:
        Tensor: A transposed tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> out = flow.transpose(input, 0, 1).shape
        >>> out
        flow.Size([6, 2, 5, 3])

    """

    x_shape = input.shape

    if dim0 < 0:
        dim0 += len(x_shape)
    if dim1 < 0:
        dim1 += len(x_shape)
    assert dim0 >= 0 and dim0 < len(x_shape), "Invalid dim0 {}, len(shape): {}".format(
        dim0, len(x_shape)
    )
    assert dim1 >= 0 and dim1 < len(x_shape), "Invalid dim1 {}, len(shape): {}".format(
        dim1, len(x_shape)
    )
    perm = []
    for i in range(len(x_shape)):
        perm.append(i)
    (perm[dim0], perm[dim1]) = (perm[dim1], perm[dim0])
    return flow.F.transpose(input, perm=perm)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
