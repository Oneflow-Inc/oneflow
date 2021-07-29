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



class IndexSelect(Module):
    def __init__(
        self, dim: int = 0, sparse_grad: bool = False,
    ):
        super.__init__()
        assert sparse_grad is False, "Only support bool = False for now!"
        self.dim = dim
    
    def forward(self, input, index):
        assert len(index.shape) == 1, "Dimensions of index should equal 1"
        assert self.dim <= len(input.shape
        ), "Value of dim is out of range(dim should be 0 or 1)"
               
        index_rshp = shape2list(input.shape)                           # shape list
        for index_i in index:
            assert index_i < index_rshp[0], \
            "value of index out of range(index shuold lower than the first dimension of input)"                    
        
        index_gather = index
      
        index_rshp[self.dim] = 1
        index_gather = index[0].expand(index_rshp)                     # reshape
        for index_i in index[1:]:        
            x=index_i.expand(index_rshp)
            index_gather = flow.cat((index_gather,x),self.dim)         # concat          

        return flow.gather(input, index_gather, self.dim)


@register_tensor_op("index_select")
def index_select_op(input, index, dim=0, sparse_grad=False):
    r"""Select values along an axis specified by `dim`.

    :attr:`index` must be an Int32 Tensor with dimension=1.
    attr:`dim` must be in the range of input Dimensions.
    value of attr:`index` must be in the range of the dim(th) of input  
    
    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (Int32Tensor): the indices of elements to select
    
    For example:

    .. code block::python
        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = np.random.randn(3, 4, 5)
        >>> index = flow.tensor([0,1], dtype=flow.int32)
        >>> output = flow.index_select(flow.Tensor(input), index, dim=1)
        >>> output.shape
        flow.Size([3, 2, 3])

    """

if __name__ == '__main__':
    
    import doctest

    doctest.testmod(raise_on_error=True)



