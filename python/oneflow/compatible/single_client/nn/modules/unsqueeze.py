from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Unsqueeze(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        assert (
            -(1 + input.ndimension()) <= self.dim <= input.ndimension()
        ), "dim should within the range [-input.ndimension() - 1, input.ndimension() + 1)"
        if self.dim < 0:
            self.dim = 1 + input.ndimension() + self.dim
        return flow.F.expand_dims(input, axis=self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
