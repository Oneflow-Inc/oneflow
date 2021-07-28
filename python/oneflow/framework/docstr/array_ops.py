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

# refer to:https://pytorch.org/docs/stable/generated/torch.diag.html 
add_docstr(
    oneflow.F.diag,
    r"""
    diag(x: Tensor, *, diagonal=0) -> Tensor

    Returns:
        oneflow.Tensor: the output Tensor.

    If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.

    If input is a matrix (2-D tensor), then returns a 1-D tensor with diagonal elements of input.
    
    Args:
        input (Tensor): the input tensor.
        diagonal (Optional[int], 0): The diagonal to consider. 
            If diagonal = 0, it is the main diagonal. 
            
            If diagonal > 0, it is above the main diagonal. 
            
            If diagonal < 0, it is below the main diagonal. Defaults to 0.
    
    For example:
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array(
        ...     [
        ...        [1.0, 2.0, 3.0],
        ...        [4.0, 5.0, 6.0],
        ...        [7.0, 8.0, 9.0],
        ...     ]
        ... )
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> flow.F.diag(input)
        tensor([1., 5., 9.], dtype=oneflow.float32)
    
    """,
)

add_docstr(
    oneflow.F.argwhere,
    r"""
    argwhere(Tensor x, *, DataType dtype=Int32) -> TensorTuple
        
    This operator finds the indices of input Tensor `x` elements that are non-zero. 
        
    Returns:
        oneflow.Tensor: The result Tensor.

    Args:
        x (oneflow.Tensor): The input Tensor.

        dtype (Optional[flow.dtype], optional): The data type of output. Defaults to None.

    For example:

    .. code-block:: python
        
        >>> import oneflow as flow
        >>> x = flow.Tensor([[0, 1, 0],
        ...                  [2, 0, 2]])
        >>> output = flow.argwhere(x)
        >>> output
        tensor([[0, 1],
                [1, 0],
                [1, 2]], dtype=oneflow.int32)

    """,
)
