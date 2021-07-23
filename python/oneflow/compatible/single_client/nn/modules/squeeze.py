from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.framework import id_util as id_util
from typing import Optional, Sequence


class Squeeze(Module):
    def __init__(self, dim: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return x
        return flow.F.squeeze(x, dim=self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
