from typing import Optional, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class To(Module):
    def __init__(self, copy):
        super().__init__()
        self.copy = copy

    def forward(self, x, device, dtype):
        result = x
        if device is not None:
            if x.device != device or self.copy:
                result = flow.F.copy(x, device_type=device.type, device_id=device.index)
        if dtype is not None:
            if x.dtype != dtype or self.copy:
                result = flow.F.cast(result, dtype=dtype)
        return result


@register_tensor_op("to")
def to_op(input, *args, **kwargs):
    """Performs Tensor dtype and/or device conversion. 
        A flow.dtype and flow.device are inferred from the arguments of `input.to(*args, **kwargs)`.
    
    .. note::
        If the ``input`` Tensor already
        has the correct :class:`flow.dtype` and :class:`flow.device`, then ``input`` is returned.
        Otherwise, the returned tensor is a copy of ``input`` with the desired.

    Args:
        input (oneflow.Tensor): An input tensor.
        *args (oneflow.Tensor or oneflow.device or oneflow.dtype): Positional arguments
        **kwargs (oneflow.device or oneflow.dtype) : Key-value arguments

    Returns:
        oneflow.Tensor: A Tensor.
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        >>> input = flow.Tensor(arr)
        >>> output = input.to(dtype=flow.float32)
        >>> np.array_equal(arr.astype(np.float32), output.numpy())
        True

    """
    copy = kwargs.get("copy", False)
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    if len(args) > 0:
        if isinstance(args[0], flow.Tensor):
            if len(args) == 2:
                copy = args[1]
            return To(copy)(input, args[0].device, args[0].dtype)
        elif isinstance(args[0], flow.dtype):
            if len(args) == 2:
                copy = args[1]
            return To(copy)(input, None, args[0])
        else:
            device = flow.device(args[0]) if isinstance(args[0], str) else args[0]
            if len(args) > 1:
                dtype = args[1]
                assert isinstance(dtype, flow.dtype)
            if len(args) > 2:
                copy = args[2]
            assert isinstance(device, flow.device)
            return To(copy)(input, device, dtype)
    if isinstance(device, flow.device) or isinstance(dtype, flow.dtype):
        return To(copy)(input, device, dtype)
    raise TypeError("to() received an invalid combination of arguments")


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
