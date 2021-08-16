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


@register_tensor_op("le")
def less_equal_op(input, other):
    """Returns the truth value of :math:`input <= other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([1, 0, 1], dtype=oneflow.int8)

    """
    return flow.F.less_equal(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
