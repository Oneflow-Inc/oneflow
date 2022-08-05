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
    oneflow._C.dropout,
    """
    dropout(x: Tensor, p: float = 0.5, training: bool = True, generator :Generator = None, *, addend: Tensor) -> Tensor 
    
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.dropout.html.

    Args:      
        x(Tensor): A Tensor which will be applyed dropout. 
        p(float): probability of an element to be zeroed. Default: 0.5    
        training(bool): If is True it will apply dropout. Default: True     
        generator(Generator, optional):  A pseudorandom number generator for sampling
        addend(Tensor, optional):  A Tensor add in result after dropout, it can be used in model's residual connection structure. Default: None  

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    Example 1: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

       
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> y = flow.nn.functional.dropout(x, p=0) 

        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> generator = flow.Generator()
        >>> y = flow.nn.functional.dropout(x, p=0.5, generator=generator) 
      
    Example 2: 
    
    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

       
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> addend = flow.ones((3, 4), dtype=flow.float32)
        >>> y = flow.nn.functional.dropout(x, p=0, addend=addend) 
        >>> y #doctest: +ELLIPSIS
        tensor([[ 0.2203,  1.2264,  1.2458,  1.4163],
                [ 1.4299,  1.3626,  0.5108,  1.4141],
                [-0.4115,  2.2183,  0.4497,  1.6520]], dtype=oneflow.float32)
    
    See :class:`~oneflow.nn.Dropout` for details.   
 
    """,
)
