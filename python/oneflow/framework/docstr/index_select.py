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
    oneflow.index_select,
    """
    input.index_select(dim, index) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchindex_select

    Select values along an axis specified by `dim`.

    :attr:`index` must be an Int32 Tensor with 1-D.
    :attr:`dim` must be in the range of input Dimensions.
    value of :attr:`index` must be in the range of the dim-th of input.
    Note that ``input`` and ``index`` do not broadcast against each other.  
    
    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (Tensor): the 1-D tensor containing the indices to index
    
    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,3],[4,5,6]], dtype=flow.int32)
        >>> input 
        tensor([[1, 2, 3],
                [4, 5, 6]], dtype=oneflow.int32)
        >>> index = flow.tensor([0,1], dtype=flow.int32)
        >>> output = flow.index_select(input, 1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
        >>> output = input.index_select(1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
    """,
)
