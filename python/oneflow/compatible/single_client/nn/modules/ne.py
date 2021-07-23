from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op


class Ne(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, other):
        if isinstance(other, flow.Tensor) or isinstance(
            other, oneflow._oneflow_internal.Tensor
        ):
            for i in range(len(input.size())):
                assert (
                    input.shape[i] >= other.shape[i]
                ), "The second tensor's shape should broadcastable with the first argument."
                if input.dtype != other.dtype:
                    other = other.to(dtype=input.dtype)
        elif isinstance(other, int) or isinstance(other, float):
            other = flow.Tensor([other], dtype=input.dtype, device=input.device)
        else:
            raise NotImplementedError(
                "Unsupport data type, The second argument can be a tensor whose shape is broadcastable with the first argument."
            )
        return flow.F.broadcast_not_equal(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
