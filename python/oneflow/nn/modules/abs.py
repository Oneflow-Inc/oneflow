import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class Abs(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.abs(x)

@register_tensor_op('abs')
def abs_op(x):
    """Return the absolute value of each element in input tensor:math:`y = |x|` element-wise.

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.Tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
        >>> flow.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)

    """
    return Abs()(x)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)