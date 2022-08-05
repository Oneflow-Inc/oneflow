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
    oneflow.argsort,
    """
    argsort() -> Tensor
    This operator sorts the input Tensor at specified dim and returns the indices of the sorted Tensor.

    Args:
        input (oneflow.Tensor): the input Tensor.
        dim (int, optional): the dimension to be sorted. Defaults to the last dim (-1).
        descending (bool, optional): controls the sorting order (ascending or descending).

    Returns:
        oneflow.Tensor: The indices of the sorted Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([[10, 2, 9, 3, 7],
        ...               [1, 9, 4, 3, 2]]).astype("float32")
        >>> input = flow.Tensor(x)
        >>> output = flow.argsort(input)
        >>> output
        tensor([[1, 3, 4, 2, 0],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, descending=True)
        >>> output
        tensor([[0, 2, 4, 3, 1],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, dim=0)
        >>> output
        tensor([[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]], dtype=oneflow.int32)

    """,
)
