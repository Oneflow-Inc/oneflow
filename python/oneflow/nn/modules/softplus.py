import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class Softplus(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.softplus(x)


@register_tensor_op("softplus")
def softplus_op(x):
    """Applies the element-wise function:

    .. math::
        Softplus(x)= \\frac{1}{β}*log(1+exp(β∗x))

    SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function when :attr:`input X β > threshold`.

    Args:
        beta:the value for the Softplus formulation.Default:1

        threshold:values above this revert to a linear function.Default:20

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x1 = flow.Tensor(np.array([1, 2, 3]))
        >>> x2 = flow.Tensor(np.array([1.53123589,0.54242598,0.15117185]))
        >>> x3 = flow.Tensor(np.array([1,0,-1]))

        >>> flow.softplus(x1).numpy()
        array([1.3132616, 2.126928 , 3.0485873], dtype=float32)
        >>> flow.softplus(x2).numpy()
        array([1.7270232, 1.0006962, 0.771587 ], dtype=float32)
        >>> flow.softplus(x3).numpy()
        array([1.3132616 , 0.6931472 , 0.31326166], dtype=float32)

    """
    return Softplus()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
