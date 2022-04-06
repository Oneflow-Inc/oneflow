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
    oneflow.tensordot,
    r"""
    tensordot(a, b, dims=Union[int, Tensor, Tuple]) -> Tensor
    
    Compute tensor dot along given dimensions.
    
    Given two tensors a and b, and dims which has two list containing dim indices, `tensordot` traverses the two 
    lists and calculate the tensor dot along every dim pair.

    Args:
        a: The input tensor to compute tensordot
        b: The input tensor to compute tensordot
        dims: int or array-like
        
    Returns:
        Oneflow.Tensor: The result tensor

    For example:
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.randn(3, 4, 5)
        >>> b = flow.randn(4, 5, 6)
        >>> flow.tensordot(a, b, dims=2).shape
        oneflow.Size([3, 6])
        >>> b = flow.randn(5, 6, 7)
        >>> flow.tensordot(a, b, dims=1).shape
        oneflow.Size([12, 42])
        >>> b = flow.randn(3, 4, 7)
        >>> flow.tensordot(a, b, dims=[[0, 1], [0, 1]]).shape
        oneflow.Size([5, 7])
    
    """,
)
