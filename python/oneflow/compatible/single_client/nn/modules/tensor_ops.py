from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.oneflow_export import experimental_api
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op

class TypeAs(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.to(dtype=target.dtype)

class Long(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.to(dtype=flow.int64)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)