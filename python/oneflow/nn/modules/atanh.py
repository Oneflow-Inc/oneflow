import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class Atanh(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.atanh(x)

def atanh_op(input):
    """Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tanh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.atanh(input)
        >>> output
        tensor([0.5493, 0.6931, 0.8673], dtype=oneflow.float32)

    """
    return Atanh()(input)

@register_tensor_op('atanh')
def atanh_op_tensor(x):
    """
    atanh() -> Tensor
    See :func:`oneflow.atanh`

    """
    return Atanh()(x)

def arctanh_op(input):
    """

    Alias for :func:`oneflow.atanh`
    """
    return Atanh()(input)

@register_tensor_op('arctanh')
def arctanh_op_tensor(input):
    """

    Alias for :func:`oneflow.atanh`
    """
    return Atanh()(input)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)