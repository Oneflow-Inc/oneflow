import random
import sys

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework import id_util as id_util
from oneflow.compatible.single_client.python.nn.module import Module


class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
