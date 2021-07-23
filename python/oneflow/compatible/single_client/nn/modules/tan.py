from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op

class Tan(Module):

    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op('tan').Input('x').Output('y').Build()

    def forward(self, x):
        return self._op(x)[0]
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)