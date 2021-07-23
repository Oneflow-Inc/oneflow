from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op


class BMM(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, mat2):
        assert (
            input.shape[0] == mat2.shape[0] and input.shape[2] == mat2.shape[1]
        ), f"batch dim or matmul dim not match, please check input!"
        return flow.F.batch_matmul(input, mat2)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
