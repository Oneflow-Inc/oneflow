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
    oneflow.topk,
    """Finds the values and indices of the k largest entries at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        k (int): the k in “top-k”
        dim (int, optional): the dimension to sort along. Defaults to the last dim (-1)
        largest (bool, optional): controls whether to return largest or smallest elements
        sorted (bool, optional): controls whether to return the elements in sorted order (Only Support True Now!)

    Returns:
        Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int32)): A tuple of (values, indices), where
        the indices are the indices of the elements in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=np.float32)
        >>> result = flow.topk(flow.Tensor(x), k=3, dim=1)
        >>> result.values
        tensor([[8., 7., 3.],
                [9., 4., 3.]], dtype=oneflow.float32)
        >>> result.indices
        tensor([[2, 3, 1],
                [1, 2, 3]], dtype=oneflow.int64)
        >>> result.values.shape
        oneflow.Size([2, 3])
        >>> result.indices.shape
        oneflow.Size([2, 3])
        >>> result = flow.topk(flow.Tensor(x), k=2, dim=1, largest=False)
        >>> result.values
        tensor([[1., 2.],
                [1., 2.]], dtype=oneflow.float32)
        >>> result.indices
        tensor([[0, 4],
                [0, 4]], dtype=oneflow.int64)
        >>> result.values.shape
        oneflow.Size([2, 2])
        >>> result.indices.shape
        oneflow.Size([2, 2])

    """,
)
