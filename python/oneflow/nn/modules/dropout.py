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
import random
import sys

import oneflow as flow
import oneflow.framework.id_util as id_util
from oneflow.nn.module import Module


class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


class Dropout(_DropoutNd):
    """During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    "Improving neural networks by preventing co-adaptation of feature
    detectors".

    Furthermore, the outputs are scaled by a factor of :math:`\\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Additionally, we can pass an extra Tensor `addend` which shape is consistent with input Tensor. 
    The `addend` Tensor will be add in result after dropout, it is very useful in model's residual connection structure.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        generator:  A pseudorandom number generator for sampling

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    example 1: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
                [ 0.4299,  0.3626, -0.4892,  0.4141],
                [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)
    
    example 2: 
    
    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout(p=0)
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> addend = flow.ones((3, 4), dtype=flow.float32)
        >>> y = m(x, addend=addend)
        >>> y #doctest: +ELLIPSIS
        tensor([[ 0.2203,  1.2264,  1.2458,  1.4163],
                [ 1.4299,  1.3626,  0.5108,  1.4141],
                [-0.4115,  2.2183,  0.4497,  1.6520]], dtype=oneflow.float32)
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, generator=None):
        _DropoutNd.__init__(self, p, inplace)
        self.p = p
        if generator is None:
            generator = flow.Generator()
        self.generator = generator

    def forward(self, x, addend=None):
        return flow._C.dropout(
            x,
            self.p,
            self.training,
            self.generator,
            addend=addend if addend is not None else None,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
