from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op

class MaskedFill(Module):

    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def forward(self, input, mask):
        in_shape = tuple(input.shape)
        value_like_x = flow.Tensor(*in_shape, device=input.device)
        value_like_x.fill_(self.value)
        return flow.F.where(mask, value_like_x, input)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)