import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class Flatten(Module):
    """Flattens a contiguous range of dims into a tensor. For use with: nn.Sequential.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    

    For example: 

    .. code-block:: python 

        >>> import oneflow as flow
        >>> input = flow.Tensor(32, 1, 5, 5)
        >>> m = flow.nn.Flatten()
        >>> output = m(input)
        >>> output.shape
        flow.Size([32, 25])

    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return flow.F.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


@register_tensor_op("flatten")
def _flow_flatten(input, start_dim: int = 0, end_dim: int = -1):
    """Flattens a contiguous range of dims into a tensor.

    Args:
        start_dim: first dim to flatten (default = 0).
        end_dim: last dim to flatten (default = -1).
    
    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.Tensor(32, 1, 5, 5)
        >>> output = input.flatten(start_dim=1)
        >>> output.shape
        flow.Size([32, 25])

    """
    return Flatten(start_dim=start_dim, end_dim=end_dim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
