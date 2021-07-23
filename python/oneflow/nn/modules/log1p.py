import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class Log1p(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("log1p").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@register_tensor_op("log1p")
def log1p_op(input):
    """Returns a new tensor with the natural logarithm of (1 + input).

    .. math::
        \\text{out}_{i}=\\log_e(1+\\text{input}_{i})

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> out = flow.log1p(x).numpy()
        >>> out
        array([0.8329091 , 0.91629076, 1.3083328 ], dtype=float32)

    """
    return Log1p()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
