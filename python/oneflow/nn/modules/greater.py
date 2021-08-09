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


def greater_op(input, other):
    """Returns the truth value of :math:`input > other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)

        >>> out = flow.gt(input1, input2).shape
        >>> out
        flow.Size([2, 6, 5, 3])

    """

    if input.dtype != flow.float32:
        input = flow.cast(input, flow.float32)
    if isinstance(other, int) or isinstance(other, float):
        other = flow.Tensor(
            [float(other)], dtype=flow.float32, device=flow.device(input.device.type)
        )
    if other.dtype != flow.float32:
        other = flow.cast(other, flow.float32)
    return flow.F.broadcast_greater(input, other)


@register_tensor_op("gt")
def greater_op_tensor(input, other):
    """

    gt() -> Tensor

    See :func:`oneflow.gt`

    """
    return greater_op(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
