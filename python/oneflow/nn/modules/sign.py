import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Sign(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.sign(x)


def sign_op(x):
    """Computes the sign of Tensor.

    .. math::

        \\text{out}_{i}  = \\text{sgn}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([-2, 0, 2]).astype(np.float32))
        >>> out1 = flow.sign(x1)
        >>> out1.numpy()
        array([-1.,  0.,  1.], dtype=float32)
        >>> x2 = flow.Tensor(np.array([-3.2, -4.5, 5.8]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.sign(x2)
        >>> out2.numpy()
        array([-1., -1.,  1.], dtype=float32)

    """
    return Sign()(x)


@register_tensor_op("sign")
def sign_op_tensor(x):
    """

    sign() -> Tensor

    See :func:`oneflow.sign`

    """
    return Sign()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
