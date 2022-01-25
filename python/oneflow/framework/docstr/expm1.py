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
    oneflow.expm1,
    """Returns a new tensor with the exponential of the elements minus 1
    of :attr:`input`.


    The equation is: 

    .. math::
        y_{i} = e^{x_{i}} - 1

    Args:
        input (oneflow.Tensor): A Tensor.
    
    Returns:
        oneflow.Tensor: The result Tensor

    For example: 

    .. code-block:: python 
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.expm1(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1.7183,  6.3891, 19.0855], dtype=oneflow.float32)


        >>> x = flow.Tensor(np.array([[2, 4, 6],[7, 8, 9]]).astype(np.float32))
        >>> y = x.expm1()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[6.3891e+00, 5.3598e+01, 4.0243e+02],
                [1.0956e+03, 2.9800e+03, 8.1021e+03]], dtype=oneflow.float32)



        >>> x = flow.Tensor(np.array([[[2, 4, 6],[7, 8, 9]],[[10,11,12],[13,14,15]]]).astype(np.float32))
        >>> y = flow.expm1(x)
        >>> print(y.shape)
        oneflow.Size([2, 2, 3])
        >>> print(y.numpy())
        [[[6.3890562e+00 5.3598152e+01 4.0242880e+02]
          [1.0956332e+03 2.9799580e+03 8.1020840e+03]]
        <BLANKLINE>
         [[2.2025465e+04 5.9873141e+04 1.6275380e+05]
          [4.4241238e+05 1.2026032e+06 3.2690165e+06]]]


    """
)
