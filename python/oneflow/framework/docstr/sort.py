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
    oneflow.sort,
    """Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Args:
        input (oneflow.Tensor): the input Tensor.
        dim (int, optional): the dimension to be sorted. Defaults to the last dim (-1).
        descending (bool, optional): controls the sorting order (ascending or descending).

    Returns:
        Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int32)): A tuple of (values, indices), where
        where the values are the sorted values and the indices are the indices of the elements
        in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=np.float32)
        >>> input = flow.Tensor(x)
        >>> result = flow.sort(input)
        >>> result.values
        tensor([[1., 2., 3., 7., 8.],
                [1., 2., 3., 4., 9.]], dtype=oneflow.float32)
        >>> result.indices
        tensor([[0, 4, 1, 3, 2],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> result = flow.sort(input, descending=True)
        >>> result.values
        tensor([[8., 7., 3., 2., 1.],
                [9., 4., 3., 2., 1.]], dtype=oneflow.float32)
        >>> result.indices
        tensor([[2, 3, 1, 4, 0],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> result = flow.sort(input, dim=0)
        >>> result.values
        tensor([[1., 3., 4., 3., 2.],
                [1., 9., 8., 7., 2.]], dtype=oneflow.float32)
        >>> result.indices
        tensor([[0, 0, 1, 1, 0],
                [1, 1, 0, 0, 1]], dtype=oneflow.int32)
 
    """,
)
