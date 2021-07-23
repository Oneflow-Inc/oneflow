from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op

class Triu(Module):

    def __init__(self, diagonal=0):
        super().__init__()
        self.diagonal = diagonal

    def forward(self, x):
        return flow.F.triu(x, self.diagonal)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)