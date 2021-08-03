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

from oneflow.framework.tensor import Tensor, register_tensor_op
from oneflow.nn.module import Module


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))

class IndexSelect(Module):
    def __init__(self, dim: int = 0, sparse_grad: bool = False):
        super().__init__()
        assert sparse_grad is False, "Only support bool = False for now!"
        self.dim = dim
    
    def forward(self, input, index):
        assert len(index.shape) == 1, "Dimensions of index should be an LongTensor"
        assert self.dim < len(input.shape), "Value of dim is out of range"       
        assert _input_args_is_int(index.tolist()), "input index parameter is not illegal!"
        
        index_rshp = list(input.shape)                           

        for index_i in index:
            assert index_i < index_rshp[0], \
            "value of index out of range(index shuold lower than the first dimension of input)"                    
        
        index_rshp[self.dim] = 1
        index_gather = index[0].expand(index_rshp)                     
        for index_i in index[1:]:        
            x=index_i.expand(index_rshp)
            index_gather = flow.cat((index_gather,x), self.dim)                   

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
        >>> input = flow.tensor(np.random.randn(3, 4, 5))
        >>> input 
        tensor([[[ 0.3769, -0.7527,  0.1159,  0.7326, -0.6883],
                 [-1.2042, -1.383 ,  2.1489, -1.8246,  1.3503],
                 [-0.9423,  0.6951,  1.5195,  0.308 ,  0.9677],
                 [-1.7696, -1.5415,  0.6982,  0.7062,  1.8302]],

                 [[-1.2   , -0.6626,  0.6486, -1.2259, -0.8657],
                 [ 0.7469, -0.3936,  0.2949, -0.0718,  2.9406],
                 [ 0.0648,  1.3443,  1.561 ,  0.4251,  0.2816],
                 [-0.3225,  1.0197, -0.3377, -0.5388,  0.0228]],

                 [[ 0.4686,  1.7773, -0.1256, -0.4089,  0.8458],
                 [-0.6285, -0.4113,  0.7034,  0.2701,  1.9672],
                 [ 0.3848,  0.9598, -0.4312,  0.8651,  0.1515],
                 [-0.0298, -1.9139, -1.1253, -0.3072, -0.0976]]],
                 dtype=oneflow.float64)
        >>> index = flow.tensor([0,1], dtype=flow.int32)
        >>> output = flow.index_select(input, 1, index)
        >>> output
        tensor([[[ 0.3769, -0.7527,  0.1159,  0.7326, -0.6883],
                 [-1.2042, -1.383 ,  2.1489, -1.8246,  1.3503]],

                 [[-1.2   , -0.6626,  0.6486, -1.2259, -0.8657],
                 [ 0.7469, -0.3936,  0.2949, -0.0718,  2.9406]],

                 [[ 0.4686,  1.7773, -0.1256, -0.4089,  0.8458],
                 [-0.6285, -0.4113,  0.7034,  0.2701,  1.9672]]],
                 dtype=oneflow.float32)

    """
    return IndexSelect(dim, sparse_grad)(input, index)

if __name__ == '__main__':
    
    import doctest

    doctest.testmod(raise_on_error=True)



