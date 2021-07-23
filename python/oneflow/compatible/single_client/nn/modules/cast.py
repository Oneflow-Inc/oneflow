from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op


class Cast(Module):
    def __init__(self, dtype: flow.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return flow.F.cast(x, dtype=self.dtype)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
