from typing import Sequence

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Reshape(Module):
    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return flow.F.reshape(x, shape=self.shape)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
