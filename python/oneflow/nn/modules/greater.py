import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class Greater(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        if x.dtype != flow.float32:
            x = flow.cast(x, flow.float32)
        if isinstance(y, int) or isinstance(y, float):
            y = flow.Tensor([float(y)], dtype=flow.float32, device=flow.device(x.device.type))
        if y.dtype != flow.float32:
            y = flow.cast(y, flow.float32)
        return flow.F.broadcast_greater(x, y)

def greater_op(x, y):
    """Returns the truth value of :math:`x > y` element-wise.

    Args:
        x (oneflow.Tensor): A Tensor
        y (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)

        >>> out = flow.gt(input1, input2).shape
        >>> out
        flow.Size([2, 6, 5, 3])

    """
    return Greater()(x, y)

@register_tensor_op('gt')
def greater_op_tensor(x, y):
    """

    gt() -> Tensor

    See :func:`oneflow.gt`

    """
    return Greater()(x, y)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)