import collections
from typing import Optional, Sequence, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.modules.utils import _check_axis


class Floor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.floor(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
