import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Less(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        if x.dtype != flow.float32:
            x = flow.cast(x, flow.float32)
        if isinstance(y, int) or isinstance(y, float):
            y = flow.Tensor(
                [float(y)], dtype=flow.float32, device=flow.device(x.device.type)
            )
        if y.dtype != flow.float32:
            y = flow.cast(y, flow.float32)
        return flow.F.broadcast_less(x, y)


@register_tensor_op("lt")
def less_op(x, y):
    """Returns the truth value of :math:`x < y` element-wise.

    Args:
        x (oneflow.Tensor): A Tensor
        y (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.array([1, 2, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.lt(input1, input2)
        >>> out
        tensor([0, 0, 1], dtype=oneflow.int8)

    """
    return Less()(x, y)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
