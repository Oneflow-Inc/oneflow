from typing import Optional
import oneflow as flow
import numpy as np
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class Argwhere(Module):

    def __init__(self, dtype) -> None:
        super().__init__()
        if dtype == None:
            dtype = flow.int32
        self.dtype = dtype

    def forward(self, x):
        (res, size) = flow.F.argwhere(x, dtype=self.dtype)
        slice_tup_list = [[0, int(size.numpy()), 1]]
        return flow.slice(res, slice_tup_list=slice_tup_list)

def argwhere_op(x, dtype: Optional[flow.dtype]=None):
    """This operator finds the indices of input Tensor `x` elements that are non-zero. 

    It returns a list in which each element is a coordinate that points to a non-zero element in the condition.

    Args:
        x (oneflow.Tensor): The input Tensor.
        dtype (Optional[flow.dtype], optional): The data type of output. Defaults to None.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([[0, 1, 0],
        ...            [2, 0, 2]]).astype(np.float32)
        
        >>> input = flow.Tensor(x)
        >>> output = flow.argwhere(input)
        >>> output
        tensor([[0, 1],
                [1, 0],
                [1, 2]], dtype=oneflow.int32)

    """
    return Argwhere(dtype=dtype)(x)

@register_tensor_op('argwhere')
def argwhere_tebsor_op(x, dtype: Optional[flow.dtype]=None):
    """

    argwhere() -> Tensor

    See :func:`oneflow.argwhere`

    """
    return Argwhere(dtype=dtype)(x)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)