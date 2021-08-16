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


@register_tensor_op("ne")
def ne_op(input, other):
    """
    Computes element-wise not equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.Tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.Tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.ne(input, other)
        >>> y
        tensor([0, 0, 0, 1], dtype=oneflow.int8)

    """
    return flow.F.not_equal(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
