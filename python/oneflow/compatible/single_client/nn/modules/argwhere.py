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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Argwhere(Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        if dtype == None:
            dtype = flow.int32
        self.dtype = dtype

    def forward(self, x):
        (res, size) = flow.F.argwhere(x, dtype=self.dtype)
        slice_tup_list = [[0, int(size.numpy()), 1]]
        return flow.experimental.slice(res, slice_tup_list=slice_tup_list)


def argwhere_op(x, dtype: Optional[flow.dtype] = None):
    """This operator finds the indices of input Tensor `x` elements that are non-zero. 

    It returns a list in which each element is a coordinate that points to a non-zero element in the condition.

    Args:
        x (oneflow.compatible.single_client.Tensor): The input Tensor.
        dtype (Optional[flow.dtype], optional): The data type of output. Defaults to None.

    Returns:
        oneflow.compatible.single_client.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([[0, 1, 0],
        ...            [2, 0, 2]]).astype(np.float32)
        
        >>> input = flow.Tensor(x)
        >>> output = flow.argwhere(input)
        >>> output
        tensor([[0, 1],
                [1, 0],
                [1, 2]], dtype=oneflow.int32)

    """
    return Argwhere(dtype=dtype)(x)


@register_tensor_op("argwhere")
def argwhere_tebsor_op(x, dtype: Optional[flow.dtype] = None):
    """

    argwhere() -> Tensor

    See :func:`oneflow.compatible.single_client.experimental.argwhere`

    """
    return Argwhere(dtype=dtype)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
