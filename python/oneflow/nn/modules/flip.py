import collections
from typing import Optional, Sequence, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_axis


class Flip(Module):
    def __init__(self, dims) -> None:
        super().__init__()
        assert isinstance(dims, (list, tuple)), f"dims must be list or tuple"
        self.dims = dims

    def forward(self, x):
        input_len = len(x.shape)
        assert (
            len(self.dims) <= input_len
        ), f"len of dims must less than len of input tensor"
        new_dims = []
        for i in self.dims:
            if i < 0:
                i += input_len
            assert (
                i < input_len
            ), f"IndexError: Dimension out of range (expected to be in range of {input_len}, but got {i})"
            new_dims.append(i)
        return flow.F.flip(x, new_dims)


def flip_op(input, dims):
    """
    
    Reverse the order of a n-D tensor along given axis in dims.

    .. note::
        `flow.flip` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flip`,
        which returns a view in constant time. Since copying a tensor's data is more work than viewing that data,
        `flow.flip` is expected to be slower than `np.flip`.

    Args:
        input (Tensor): the input tensor
        dims (a list or tuple): axis to flip on
        
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> np_arr = np.arange(0, 8).reshape((2, 2, 2)).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> out = flow.flip(input, [0, 1])
        >>> out
        tensor([[[6., 7.],
                 [4., 5.]],
        <BLANKLINE>
                [[2., 3.],
                 [0., 1.]]], dtype=oneflow.float32)

    """
    return Flip(dims)(input)


@register_tensor_op("flip")
def flip_op_tensor(input, dims):
    """
    See :func:`oneflow.flip`
    """
    return Flip(dims)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
