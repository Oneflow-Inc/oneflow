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
    oneflow.amin,
    """
    amin(input, dim, keepdim=False) -> Tensor  
    
    Returns the minimum value of each slice of the `input` tensor in the given dimension(s) `dim`.

    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed (see :func:`oneflow.squeeze`), resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).
    
    This function is equivalent to PyTorch’s amin function. 
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.amin.html.

    Parameters:
        input (oneflow.Tensor): the input Tensor.
        dim (int, Tuple[int]): the dimension or dimensions to reduce. 
        keepdim (bool): whether the output tensor has `dim` retained or not.
    
    Example:

    .. code-block:: python

        >>> import oneflow as flow
               
        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> flow.amin(x, 1)
        tensor([[0, 1],
                [4, 5]], dtype=oneflow.int64)
        >>> flow.amin(x, 0)
        tensor([[0, 1],
                [2, 3]], dtype=oneflow.int64)
        >>> flow.amin(x)
        tensor(0, dtype=oneflow.int64)
        >>> flow.amin(x, 0, True)
        tensor([[[0, 1],
                 [2, 3]]], dtype=oneflow.int64)
    """,
)
