import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Round(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.round(x)


def round_op(x):
    """This operator rounds the value of Blob to the nearest integer.
    Args:
        x (oneflow.Tensor): A Tensor
    Returns:
        oneflow.Tensor: The result Tensor
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([1.49999, 1.500001, 2.7]).astype(np.float32))
        >>> out1 = flow.round(x1)
        >>> out1.numpy()
        array([1., 2., 3.], dtype=float32)
        >>> x2 = flow.Tensor(np.array([2.499999, 7.5000001, 5.3, 6.8]).astype(np.float32))
        >>> out2 = flow.round(x2)
        >>> out2.numpy()
        array([2., 8., 5., 7.], dtype=float32)

    """
    return Round()(x)


@register_tensor_op("round")
def round_op_tensor(x):
    """
    round() -> Tensor

    See :func:`oneflow.round`

    """
    return Round()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
