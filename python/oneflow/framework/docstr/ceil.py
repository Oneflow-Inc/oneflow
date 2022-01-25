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
    oneflow.ceil,
    """Returns a new tensor with the ceil of the elements of :attr:`input`,
    the smallest integer greater than or equal to each element.

    The equation is: 

    .. math::
        \\text{out}_{i} = \\left\\lceil \\text{input}_{i} \\right\\rceil = \\left\\lfloor \\text{input}_{i} \\right\\rfloor + 1

    Args:
        input (oneflow.Tensor): A Tensor.
    
    Returns:
        oneflow.Tensor: The result Tensor

    For example: 


    .. code-block:: python 
        
        >>> import oneflow as flow
        >>> import numpy as np   
        >>> x = flow.Tensor(np.array([0.1, -2, 3.4]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1., -2.,  4.], dtype=oneflow.float32)
        >>> x = flow.Tensor(np.array([[2.5, 4.6, 0.6],[7.8, 8.3, 9.2]]).astype(np.float32))
        >>> y = x.ceil()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[ 3.,  5.,  1.],
                [ 8.,  9., 10.]], dtype=oneflow.float32)
        >>> x = flow.Tensor(np.array([[[2.2, 4.4, 6.5],[7.1, 8.2, 9.3]],[[10.6,11.2,12.2],[13.5,14.8,15.9]]]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([2, 2, 3])
        >>> y
        tensor([[[ 3.,  5.,  7.],
                 [ 8.,  9., 10.]],
        <BLANKLINE>
                [[11., 12., 13.],
                 [14., 15., 16.]]], dtype=oneflow.float32)

    """
)
