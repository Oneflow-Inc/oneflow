from typing import Optional, Sequence

import oneflow as flow
import oneflow.framework.id_util as id_util
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class MatMul(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        assert len(a.shape) >= 2, "Tensor a's dim should >=2"
        assert len(b.shape) >= 2, "Tensor b's dim should >=2"
        if len(a.shape) == len(b.shape):
            if len(a.shape) == 2:
                res = flow.F.matmul(a, b)
            else:
                res = flow.F.batch_matmul(a, b)
        else:
            assert (
                len(b.shape) == 2
            ), "Not support number of dimensions of a being less than number of dimensions of b!"
            res = flow.F.broadcast_matmul(a, b)
        return res


@register_tensor_op("matmul")
def matmul_op(input, other):
    """This operator applies matrix multiplication to two Tensor.

    Args:
        a (oneflow.Tensor): A Tensor
        b (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input1 = flow.Tensor(np.random.randn(2, 6), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.random.randn(6, 5), dtype=flow.float32)
        >>> of_out = flow.matmul(input1, input2)
        >>> of_out.shape
        flow.Size([2, 5])

    """
    return MatMul()(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
