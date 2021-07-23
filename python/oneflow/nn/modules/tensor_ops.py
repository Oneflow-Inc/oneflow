import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op

class TypeAs(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.to(dtype=target.dtype)

@register_tensor_op('type_as')
def type_as_op(input, target):
    """Returns this tensor cast to the type of the given tensor.
        This is a no-op if the tensor is already of the correct type.

    Args:
        input  (Tensor): the input tensor.
        target (Tensor): the tensor which has the desired type.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.Tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> target = flow.Tensor(np.random.randn(4, 5, 6), dtype = flow.int32)
        >>> input = input.type_as(target)
        >>> input.dtype
        oneflow.int32

    """
    return TypeAs()(input, target)

class Long(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.to(dtype=flow.int64)

@register_tensor_op('long')
def long_op(input):
    """`Tensor.long()` is equivalent to `Tensor.to(flow.int64)`. See to().

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.Tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.long()
        >>> input.dtype
        oneflow.int64

    """
    return Long()(input)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)