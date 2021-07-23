import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op
import oneflow.framework.id_util as id_util
from typing import Optional, Sequence


class Squeeze(Module):
    def __init__(self, dim: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return x
        return flow.F.squeeze(x, dim=self.dim)


@register_tensor_op("squeeze")
def squeeze_op(input, dim: Optional[Sequence[int]] = None):
    """This operator removes the specified dimention which size is 1 of the input Tensor.
    If the `dim` is not specified, this operator will remove all the dimention which size is 1 of the input Tensor.

    The amount of element in return value is the same as Tensor `input`.

    Args:
        input (oneflow.Tensor): The input Tensor.
        dim (Optional[Sequence[int]]): The dim. Defaults to None.

    Returns:
        Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([[[[1, 1, 1]]]]).astype(np.int32))
        >>> out = flow.squeeze(input, dim=[1, 2]).shape
        >>> out
        flow.Size([1, 3])

    """
    if isinstance(dim, int):
        dim = [dim]
    elif dim is None:
        dim = range(input.ndim)
    dim = list(filter(lambda i: input.size(i) == 1, dim))
    return Squeeze(dim=dim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
