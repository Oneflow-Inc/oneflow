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
    oneflow.flatten,
    """Flattens a contiguous range of dims into a tensor.

    Args:
        start_dim: first dim to flatten (default = 0).
        end_dim: last dim to flatten (default = -1).
    
    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.randn(32, 1, 5, 5)
        >>> output = flow.flatten(input, start_dim=1)
        >>> output.shape
        oneflow.Size([32, 25])

    """,
)

add_docstr(
    oneflow.unflatten,
    """Expands a dimension of the input tensor over multiple dimensions.

    See also :func:`oneflow.flatten` the inverse of this function. It coalesces several dimensions into one.

    Args:
        input(Tensor): the input tensor.
        dim (int): Dimension to be unflattened, specified as an index into
            ``input.shape``.
        sizes (Tuple[int]): New shape of the unflattened dimension.
            One of its elements can be `-1` in which case the corresponding output
            dimension is inferred. Otherwise, the product of ``sizes`` *must*
            equal ``input.shape[dim]``.

    Returns:
        A View of input with the specified dimension unflattened.

    For example: 

    .. code-block:: python 

        >>> import oneflow as flow
        >>> input = flow.randn(5, 12, 3)
        >>> output = flow.unflatten(input, 1, (2, 2, 3, 1, 1))
        >>> output.shape
        oneflow.Size([5, 2, 2, 3, 1, 1, 3])
        >>> input = flow.randn(3, 4, 1)
        >>> output = flow.unflatten(input, 1, (2, 2))
        >>> output.shape
        oneflow.Size([3, 2, 2, 1])

    """,
)