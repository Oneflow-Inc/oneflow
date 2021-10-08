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


@register_tensor_op("squeeze")
def squeeze_op(input, dim: Optional[Sequence[int]] = None):
    """This operator removes the specified dimention which size is 1 of the input Tensor.
    If the `dim` is not specified, this operator will remove all the dimention which size is 1 of the input Tensor.

    The amount of element in return value is the same as Tensor `input`.

    Args:
        input (oneflow.Tensor): The input Tensor.
        dim (Optional[Sequence[int]]): The dim. Defaults to None.

    Returns:
        Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([[[[1, 1, 1]]]]).astype(np.int32))
        >>> input.shape
        oneflow.Size([1, 1, 1, 3])
        >>> out = flow.squeeze(input, dim=[1, 2]).shape
        >>> out
        oneflow.Size([1, 3])

    """
    if isinstance(dim, int):
        dim = [dim]
    elif dim is None:
        dim = range(input.ndim)
    dim = list(filter(lambda i: input.size(i) == 1, dim))

    return flow._C.squeeze(input, dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
