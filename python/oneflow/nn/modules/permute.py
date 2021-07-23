import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import register_tensor_op
from typing import Optional, Sequence


class Permute(Module):
    def __init__(self, *dims) -> None:
        super().__init__()
        self.perm = list(*dims)

    def forward(self, x):
        assert len(self.perm) == len(x.shape)
        new_perm = []
        for dim in self.perm:
            if dim < 0:
                dim += len(self.perm)
            assert dim >= 0 and dim < len(
                x.shape
            ), "Invalid dim0 {}, len(shape): {}".format(dim, len(x.shape))
            new_perm.append(dim)
        return flow.F.transpose(x, perm=new_perm)


@register_tensor_op("permute")
def permute_op(tensor, *dims):
    """Returns a view of the original tensor with its dimensions permuted.

    Args:
        *dims (int...): The desired ordering of dimensions

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> out = input.permute(1, 0, 2, 3).shape
        >>> out
        flow.Size([6, 2, 5, 3])

    """
    return Permute(dims)(tensor)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
