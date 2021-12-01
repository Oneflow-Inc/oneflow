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


@register_tensor_op("expand")
def expand_op(input, *sizes):
    """This operator expand the input tensor to a larger size.

    Passing -1 as the size for a dimension means not changing the size of that dimension.

    Tensor can be also expanded to a larger number of dimensions and the new ones will be appended at the front.

    For the new dimensions, the size cannot be set to -1.

    Args:
        input (oneflow.Tensor): The input Tensor.
        *sizes  (oneflow.Size or int): The desired expanded size.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[[[0, 1]],
        ...               [[2, 3]],
        ...               [[4, 5]]]]).astype(np.int32)
        >>> input = flow.Tensor(x)
        >>> input.shape
        oneflow.Size([1, 3, 1, 2])
        >>> out = input.expand(1, 3, 2, 2)
        >>> out.shape
        oneflow.Size([1, 3, 2, 2])

    """
    return flow._C.expand(input, sizes)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
