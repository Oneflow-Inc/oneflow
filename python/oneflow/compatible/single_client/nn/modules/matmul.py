from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.framework import id_util as id_util
from typing import Optional, Sequence


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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
