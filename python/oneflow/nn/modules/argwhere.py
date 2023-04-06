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
from typing import Optional

import numpy as np

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


def argwhere_op(input, dtype: Optional[flow.dtype] = flow.int32):
    """This operator finds the indices of input Tensor `input` elements that are non-zero. 

    It returns a list in which each element is a coordinate that points to a non-zero element in the condition.

    Args:
        input (oneflow.Tensor): the input Tensor.
        dtype (Optional[flow.dtype], optional): The data type of output. Defaults to None.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([[0, 1, 0],
        ...            [2, 0, 2]]).astype(np.float32)

        >>> input = flow.Tensor(x)
        >>> output = flow.argwhere(input)
        >>> output
        tensor([[0, 1],
                [1, 0],
                [1, 2]], dtype=oneflow.int32)

    """

    if input.is_lazy:
        raise ValueError("A lazy tensor can not be applied to argwhere.")

    (res, size) = flow._C.argwhere(input, dtype=dtype)
    slice_tup_list = [(0, size.numpy().item(), 1)]
    return flow.slice(res, slice_tup_list=slice_tup_list)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
