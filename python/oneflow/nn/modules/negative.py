import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class Negative(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.negative(x)

@register_tensor_op('negative')
def negative_op(x):
    """This operator computes the negative value of Tensor.

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.Tensor(
        ...    np.array([1.0, -1.0, 2.3]).astype(np.float32), dtype=flow.float32
        ... )
        >>> out = flow.negative(input)
        >>> out
        tensor([-1. ,  1. , -2.3], dtype=oneflow.float32)

    """
    return Negative()(x)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)