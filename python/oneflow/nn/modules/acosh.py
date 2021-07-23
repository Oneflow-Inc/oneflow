import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Acosh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.acosh(x)


def acosh_op(x):
    """Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

    .. math::

        \\text{out}_{i} = \\cosh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> out1 = flow.acosh(x1)
        >>> out1
        tensor([1.317 , 1.7627, 2.0634], dtype=oneflow.float32)
        >>> x2 = flow.Tensor(np.array([1.5, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.acosh(x2)
        >>> out2
        tensor([0.9624, 1.6094, 1.9827], device='cuda:0', dtype=oneflow.float32)

    """
    return Acosh()(x)


@register_tensor_op("acosh")
def acosh_op_tensor(x):
    """

    acosh() -> Tensor

    See :func:`oneflow.acosh`

    """
    return Acosh()(x)


def arccosh_op(x):
    """

    See :func:`oneflow.acosh`

    """
    return Acosh()(x)


@register_tensor_op("arccosh")
def arccosh_op_tensor(x):
    """

    arccosh() -> Tensor

    See :func:`oneflow.acosh`

    """
    return Acosh()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
