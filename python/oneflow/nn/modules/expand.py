from typing import Optional

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Expand(Module):
    def __init__(self, *sizes) -> None:
        super().__init__()
        self.expand_size = list(*sizes)

    def forward(self, x):
        if x.dtype == flow.int8:
            x = flow.cast(x, flow.int32)
        return flow.F.expand(x, self.expand_size)


@register_tensor_op("expand")
def expand_op(x, *sizes):
    """This operator expand the input tensor to a larger size.

    Passing -1 as the size for a dimension means not changing the size of that dimension.

    Tensor can be also expanded to a larger number of dimensions and the new ones will be appended at the front.

    For the new dimensions, the size cannot be set to -1.

    Args:
        x (oneflow.Tensor): The input Tensor.
        *sizes  (flow.Size or int): The desired expanded size.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[[[0, 1]],
        ...               [[2, 3]],
        ...               [[4, 5]]]]).astype(np.int32)

        >>> input = flow.Tensor(x)

        >>> out = input.expand(1, 3, 2, 2)
        >>> out.shape
        flow.Size([1, 3, 2, 2])

    """
    return Expand(sizes)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
