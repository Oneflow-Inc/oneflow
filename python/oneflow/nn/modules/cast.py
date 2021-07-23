import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Cast(Module):
    def __init__(self, dtype: flow.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return flow.F.cast(x, dtype=self.dtype)


@register_tensor_op("cast")
def cast_op(x, dtype):
    """The operation takes input tensor `x` and casts it to the output with `dtype`

    Args:
        x (oneflow.Tensor): A Tensor
        dtype (flow.dtype): Data type of the output tensor

    Returns:
        oneflow.Tensor: A Tensor with specific dtype.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input = flow.Tensor(np_arr, dtype=flow.float32)
        >>> output = flow.cast(input, flow.int8)
        >>> np.array_equal(output.numpy(), np_arr.astype(np.int8))
        True

    """
    return Cast(dtype)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
