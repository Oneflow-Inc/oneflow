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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.narrow,
    r"""
    narrow(x, dim: int, start: int, length: int) -> Tensor
    
    Returns a new tensor that is a narrowed version of `input` tensor.
    The dimension `dim` is input from `start` to `start + length`.

    Args:
        input: the tensor to narrow.
        dim: the dimension along which to narrow.
        start: the starting dimension.
        length: the distance to the ending dimension.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> flow.narrow(input, 0, 0, 2)
        tensor([[1, 2, 3],
                [4, 5, 6]], dtype=oneflow.int64)
        >>> flow.narrow(input, 1, 1, 2)
        tensor([[2, 3],
                [5, 6],
                [8, 9]], dtype=oneflow.int64)
    """,
)

add_docstr(
    oneflow.unsqueeze,
    r"""
    unsqueeze(input, dim) -> Tensor
    
    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range `[-input.ndimension() - 1, input.ndimension() + 1)`
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.ndimension() + 1``.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.randn(2, 3, 4)
        >>> y = x.unsqueeze(2)
        >>> y.shape
        oneflow.Size([2, 3, 1, 4])
    """,
)

add_docstr(
    oneflow.permute,
    r"""
    permute(input, *dims) -> Tensor

    Returns a view of the original tensor with its dimensions permuted.

    Args:
        dims (tuple of ints): The desired ordering of dimensions

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> output = flow.permute(input, (1, 0, 2, 3)).shape
        >>> output
        oneflow.Size([6, 2, 5, 3])

    """,
)
