from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module


class MeshGrid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        size = len(inputs)
        assert size > 0, f"meshgrid expects a non-empty TensorList"
        shape = list()
        for i in range(size):
            assert inputs[i].dim() <= 1, f(
                "Expected scalar or 1D tensor in the tensor list but got: ", inputs[i]
            )
            if inputs[i].dim() == 0:
                shape.append(1)
            else:
                shape.append(inputs[i].shape[0])
        for i in range(size - 1):
            assert (
                inputs[i].dtype == inputs[i + 1].dtype
                and inputs[i].device == inputs[i + 1].device
            ), f"meshgrid expects all tensors to have the same dtype and device"
        outputs = []
        for i in range(size):
            view_shape = [1] * size
            view_shape[i] = -1
            outputs.append(inputs[i].reshape(view_shape).expand(*shape))
        return outputs


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
