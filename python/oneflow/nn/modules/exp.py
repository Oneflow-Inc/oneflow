import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class Exp(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.exp(x)

@register_tensor_op('exp')
def exp_op(x):
    """This operator computes the exponential of Tensor.

    The equation is:

    .. math::

        out = e^x

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = x.exp()
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

    """
    return Exp()(x)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)