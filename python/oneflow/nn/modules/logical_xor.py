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


def logical_xor_op(input, other):
    """
    Computes the element-wise logical XOR of the given input tensors. 
    Zeros are treated as False and nonzeros are treated as True.

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute XOR with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.Tensor(np.array([1, 0, 1]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.array([1, 0, 0]).astype(np.float32), dtype=flow.float32)
        >>> out = flow.logical_xor(input1, input2)
        >>> out
        tensor([0, 0, 1], dtype=oneflow.int8)

    """
    assert input.shape == other.shape, "shape of input and other should be same"

    if other.dtype != input.dtype:
        other = flow.cast(other, input.dtype)
    return flow.F.broadcast_logical_xor(input, other)


@register_tensor_op("logical_xor")
def logical_xor_op_tensor(input, other):
    """
    logical_xor() -> Tensor

    See :func:`oneflow.logical_xor`

    """
    return logical_xor_op(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
