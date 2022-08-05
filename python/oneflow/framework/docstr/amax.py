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
    oneflow.amax,
    """
    oneflow.amax(input, dim=None, keepdim=False) -> Tensor

    Returns the maximum along a dimension.

    This function is equivalent to PyTorch’s amax function. 

    Args:
        input (oneflow.Tensor): the input Tensor.
        dim (int or List of int, optional): the dimension or the dimensions to reduce. Dim is None by default. 
        keepdim (bool, optional): whether to retain the dimension. keepdim is False by default. 

    Returns:
        oneflow.Tensor: Maximum of the input tensor

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
               
        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> flow.amax(x, 1)
        tensor([[2, 3],
                [6, 7]], dtype=oneflow.int64)
        >>> flow.amax(x, 0)
        tensor([[4, 5],
                [6, 7]], dtype=oneflow.int64)
        >>> flow.amax(x)
        tensor(7, dtype=oneflow.int64)
        >>> flow.amax(x, 0, True)
        tensor([[[4, 5],
                 [6, 7]]], dtype=oneflow.int64)
    """,
)
