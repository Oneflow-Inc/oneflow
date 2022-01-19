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
    dropout(x: Tensor, p:float = 0.5, generator :Generator = None) -> Tensor 
    


    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html

    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.


    Description of Parameter misalignment:

    Parameter generator : oneflow.nn.functional.dropout have it but torch.nn.functional.dropout do not.
    
    Parameter training : torch.nn.functional.dropout have it but oneflow.nn.functional.dropout do not.

    Args:      
        p: (float)probability of an element to be zeroed. Default: 0.5        
        generator(Generator, optional):  a pseudorandom number generator for sampling
    
        

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

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
      

    
    See :class:`~oneflow.nn.Dropout` for details.   
 
    """,
)
