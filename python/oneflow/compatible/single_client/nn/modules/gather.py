from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import Tensor
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module
from typing import Optional, List, Tuple


class Gather(Module):
    def __init__(self, dim: int = 0, sparse_grad: bool = False):
        super().__init__()
        assert sparse_grad is False, "Only support bool = False for now!"
        self.dim = dim

    def forward(self, input, index):
        assert self.dim < len(
            index.shape
        ), "Value of dim is out of range(dim should be less than len(index.shape))"
        assert len(input.shape) == len(
            index.shape
        ), "Dimensions of input and index should equal"
        for i in range(0, len(input.shape)):
            if self.dim == i:
                continue
            else:
                assert (
                    input.shape[i] == index.shape[i]
                ), "Dimensions of input and index should be same except at dim"
        return flow.F.dim_gather(input, index, dim=self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
