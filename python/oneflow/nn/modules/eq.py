import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op


class Eq(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        if isinstance(other, flow.Tensor) or isinstance(
            other, flow._oneflow_internal.Tensor
        ):
            for i in range(len(input.size())):
                assert (
                    input.shape[i] >= other.shape[i]
                ), "The second tensor's shape should broadcastable with the first argument."
                if input.dtype != other.dtype:
                    other = other.to(dtype=input.dtype)
        elif isinstance(other, int) or isinstance(other, float):
            other = flow.Tensor([other], dtype=input.dtype, device=input.device)
        else:
            raise NotImplementedError(
                "Unsupport data type, The second argument can be a tensor whose shape is broadcastable with the first argument."
            )
        return flow.F.broadcast_equal(input, other)


@register_tensor_op("eq")
def eq_op(input, other):
    """
    Computes element-wise equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.Tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.Tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.eq(input, other)
        >>> y
        tensor([1, 1, 1, 0], dtype=oneflow.int8)

    """
    return Eq()(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
