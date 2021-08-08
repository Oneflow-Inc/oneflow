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


def diag_op(input, diagonal=0):
    """
    If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
    If input is a matrix (2-D tensor), then returns a 1-D tensor with diagonal elements of input.

    Args:
        input (Tensor): the input tensor.
        diagonal (Optional[int], 0): The diagonal to consider. 
            If diagonal = 0, it is the main diagonal. If diagonal > 0, it is above the main diagonal. If diagonal < 0, it is below the main diagonal. Defaults to 0.
    
    Returns:
        oneflow.Tensor: the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array(
        ...     [
        ...        [1.0, 2.0, 3.0],
        ...        [4.0, 5.0, 6.0],
        ...        [7.0, 8.0, 9.0],
        ...     ]
        ... )

        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> flow.diag(input)
        tensor([1., 5., 9.], dtype=oneflow.float32)
    """
    return flow.F.diag(input, diagonal)


@register_tensor_op("diag")
def diag_op_tensor(input, diagonal=0):
    """
    diag() -> Tensor
    See :func:`oneflow.diag`
    
    """
    return flow.F.diag(input, diagonal)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
