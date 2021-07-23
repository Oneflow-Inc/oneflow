from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Atan2(Module):
    def __init__(self) -> None:
        super().__init__()
        self.atan2_op = (
            flow.builtin_op("atan2").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, x, y):
        return self.atan2_op(x, y)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
