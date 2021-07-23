from oneflow.compatible.single_client.python.framework.tensor import Tensor
from oneflow.compatible.single_client.python.nn.module import Module

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
