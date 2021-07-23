from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op


class Abs(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.abs(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
