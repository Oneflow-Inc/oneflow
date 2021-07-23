from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
