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

import oneflow as flow
from oneflow.framework.tensor import Tensor, register_tensor_op
from oneflow.nn.module import Module

#import numpy as np

def shape2list(args):
    out = []
    for x in args:
        out.append(x)
    return out

def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


class IndexSelect(Module):
    def __init__(self, dim: int = 0, sparse_grad: bool = False):
        super().__init__()
        assert sparse_grad is False, "Only support bool = False for now!"
        self.dim = dim
    
    def forward(self, input, index):
        assert len(index.shape) == 1, "Dimensions of index should be an LongTensor"
        assert self.dim < len(input.shape
        ), "Value of dim is out of range"
               
        index_rshp = shape2list(input.shape)                           # shape to list
        assert _input_args_is_int(index.tolist()), "input sizes parameter is not illegal!"

        for index_i in index:
            assert index_i < index_rshp[0], \
            "value of index out of range(index shuold lower than the first dimension of input)"                    
        
        index_rshp[self.dim] = 1
        index_gather = index[0].expand(index_rshp)                     # reshape
        for index_i in index[1:]:        
            x=index_i.expand(index_rshp)
            index_gather = flow.cat((index_gather,x),self.dim)         # concat          

        return flow.gather(input, index_gather, self.dim)


@register_tensor_op("index_select")
def index_select_op(input, dim, index, sparse_grad=False):
    r"""Select values along an axis specified by `dim`.

    :attr:`index` must be an Int32 Tensor with dimension=1.
    attr:`dim` must be in the range of input Dimensions.
    value of attr:`index` must be in the range of the dim(th) of input  
    
    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to select
    
    For example:

    .. code block::python
        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = np.random.randn(3, 4, 5)
        >>> index = flow.tensor([0,1], dtype=flow.int32)
        >>> output = flow.index_select(flow.Tensor(input), dim=1, index)
        >>> output.shape
        flow.Size([3, 2, 3])

    """
    return IndexSelect(dim, sparse_grad)(input, index)

if __name__ == '__main__':
    
    import doctest

    doctest.testmod(raise_on_error=True)



