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


def tril_op(input, diagonal=0):
    """Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices input along the specified diagonal, 
    the other elements of the result tensor out are set to 0.
    
    .. note::
        if diagonal = 0, the diagonal of the returned tensor will be the main diagonal,
        if diagonal > 0, the diagonal of the returned tensor will be above the main diagonal, 
        if diagonal < 0, the diagonal of the returned tensor will be below the main diagonal.

    Args:
        input (Tensor): the input tensor. 
        diagonal (int, optional): the diagonal to specify. 

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.Tensor(np.ones(shape=(3, 3)).astype(np.float32))
        >>> flow.tril(x)
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]], dtype=oneflow.float32)

    """
    return flow.F.tril(input, diagonal)


@register_tensor_op("tril")
def tril_op_tensor(input, diagonal=0):
    """
    input.tril(diagonal) -> Tensor
    See :func:`oneflow.tril`
    """

    return tril_op(input, diagonal)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
