import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class Unsqueeze(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        assert (
            -(1 + input.ndimension()) <= self.dim <= input.ndimension()
        ), "dim should within the range [-input.ndimension() - 1, input.ndimension() + 1)"
        if self.dim < 0:
            self.dim = 1 + input.ndimension() + self.dim
        return flow.F.expand_dims(input, axis=self.dim)


@register_tensor_op("unsqueeze")
def unsqueeze_op(input, dim):
    """Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range `[-input.ndimension() - 1, input.ndimension() + 1)`
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.ndimension() + 1``.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.random.rand(2, 3, 4))
        >>> y = x.unsqueeze(2)
        >>> y.shape
        flow.Size([2, 3, 1, 4])
    """
    return Unsqueeze(dim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
