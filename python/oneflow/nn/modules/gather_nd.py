import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.module import Module

class Gather_nd(Module):

    def __init__(self) -> None:
        super().__init__()
        self.gather_nd_op = flow.builtin_op('gather_nd').Input('params').Input('indices').Output('out').Build()

    def forward(self, input, index):
        return self.gather_nd_op(input, index)[0]

def gather_nd_op(input, index):
    """This operator is a high-dimensional extension of `gather`, `index` is a K-dimensional
    tensor, which is regarded as a index of input Tensor `input`.

    Each element defines a slice of `input`:

    .. math::

        output[i_{0},i_{1},...,i_{K-2}] = input[index(i_{0},i_{1},...,i_{K-2})]


    Args:
        input: The input Tensor.
        index: The slice indices.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([[1, 2,3], [4, 5,6],[7,8,9]]), dtype=flow.float)
        >>> index_1 = flow.Tensor(np.array([[0], [2]]), dtype=flow.int)
        >>> out_1 = flow.gather_nd(input,index_1)
        >>> print(out_1.shape)
        flow.Size([2, 3])
        >>> out_1
        tensor([[1., 2., 3.],
                [7., 8., 9.]], dtype=oneflow.float32)
        >>> index_2 = flow.Tensor(np.array([[0,2], [2,1]]), dtype=flow.int)
        >>> out_2 = flow.gather_nd(input,index_2)
        >>> out_2
        tensor([3., 8.], dtype=oneflow.float32)

    """
    return Gather_nd()(input, index)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)