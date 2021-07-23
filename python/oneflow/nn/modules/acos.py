import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class Acos(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.acos(x)


@register_tensor_op("acos")
def acos_op(tensor):
    """
    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\arccos(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)

    """
    return Acos()(tensor)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
