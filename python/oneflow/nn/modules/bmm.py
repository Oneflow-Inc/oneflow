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
from oneflow.framework.tensor import register_tensor_op


def bmm_op(input, mat2):
    """
    Performs a batch matrix-matrix product of matrices stored in input and mat2.

    `input` and `mat2` must be 3-D tensors each containing the same number of matrices.

    If input is a (b x n x m) tensor, mat2 is a (b x m x p) tensor, out will be a (b x n x p) tensor.

    Args:
        input(oneflow.Tensor):  the first batch of matrices to be multiplied
        mat2(oneflow.Tensor): the second batch of matrices to be multiplied
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input1 = flow.Tensor(np.random.randn(10, 3, 4))
        >>> input2 = flow.Tensor(np.random.randn(10, 4, 5))
        >>> of_out = flow.bmm(input1, input2)
        >>> of_out.shape
        oneflow.Size([10, 3, 5])
    """
    assert (
        input.shape[0] == mat2.shape[0] and input.shape[2] == mat2.shape[1]
    ), f"batch dim or matmul dim not match, please check input!"
    return flow._C.batch_matmul(input, mat2)


@register_tensor_op("bmm")
def bmm_op_tensor(input, mat2):
    """

    bmm() -> Tensor

    See :func:`oneflow.bmm`

    """
    return flow._C.batch_matmul(input, mat2)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
