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
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op

def diagonal_transpose(x, dim_num, dim1=0, dim2=1):
    outindex = [dim1, dim2]
    for i in range(dim_num):
        if i != dim1 and i != dim2:
            outindex.append(i)
    return x, outindex

def diagonal_op(input, offset=0, dim1=0, dim2=1):
    """
    Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 
    appended as a dimension at the end of the shape.
    Args:
        input (Tensor): the input tensor.Must be at least 2-dimensional.
        offset (Optional[int], 0): which diagonal to consider. Default: 0 (main diagonal)
        dim1 (Optional[int], 0): first dimension with respect to which to take diagonal. Default: 0
        dim2 (Optional[int], 1): second dimension with respect to which to take diagonal. Default: 1
    
    Returns:
        oneflow.Tensor: the output Tensor.
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
        >>> flow.diagonal(input)
        tensor([2., 6.], dtype=oneflow.float32)
    """
    dim_num = len(input.shape)
    assert (
        dim1 >= -dim_num and dim1 < dim_num
    ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
        -dim_num, dim_num - 1, dim1
    )
    assert (
        dim2 >= -dim_num and dim2 < dim_num
    ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
        -dim_num, dim_num - 1, dim1
    )
    if dim1 < 0:
        dim1 += dim_num
    if dim2 < 0:
        dim2 += dim_num
    assert dim1 != dim2, "diagonal dimensions cannot be identical {}, {}".format(
        dim1, dim2
    )
    input, outindex = diagonal_transpose(input, dim_num, dim1, dim2)
    input = flow.F.transpose(input, perm=outindex)
    out = flow.F.diagonal(input, offset)
    return out

@register_tensor_op("diagonal")
def diagonal_op_tensor(input, offset=0, dim1=0, dim2=1):
    """
    diagonal() -> Tensor
    See :func:`oneflow.diagonal`
    
    """
    dim_num = len(input.shape)