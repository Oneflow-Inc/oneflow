import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Tan(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("tan").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


def tan_op(input):
    """Returns  the tan value of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tan(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([-1/4*np.pi, 0, 1/4*np.pi]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.tan(input)
        >>> output
        tensor([-1.,  0.,  1.], dtype=oneflow.float32)

    """
    return Tan()(input)


@register_tensor_op("tan")
def tan_op_tensor(input):
    """
    tan() -> Tensor
    See :func:`oneflow.tan`

    """
    return Tan()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
