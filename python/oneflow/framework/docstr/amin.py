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
    """This function is equivalent to PyTorch’s amin function. 
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.amin.html.
    Returns the minimum value of each slice of the `input` tensor in the given dimension(s) `dim`.
    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed (see :func:`oneflow.squeeze`), resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).
    
    Args:
        input (oneflow.Tensor): the input Tensor.
        dim (int or tuple of python:ints): the dimension or dimensions to reduce. 
        keepdim (bool): whether the output tensor has `dim` retained or not.
    
    Returns:
        oneflow.Tensor: Maximum of the input tensor
    
    Example:
    .. code-block:: python
    
        >>> import oneflow as flow
        >>> a = flow.randn(4, 4)
        >>> a
        tensor([[-0.2492, -0.5142, -1.3223, -0.0793],
                [ 1.5462,  0.2256, -0.6901, -0.1943],
                [ 0.2688,  0.7676, -0.1927, -0.8397],
                [-0.0454,  0.5621,  0.1304, -1.2015]], dtype=oneflow.float32)
        >>> flow.amin(a, 1)
        tensor([-1.3223, -0.6901, -0.8397, -1.2015], dtype=oneflow.float32)
        
    """,
)
