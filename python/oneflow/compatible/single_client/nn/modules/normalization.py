import oneflow
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn import init
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import Tensor
from typing import Tuple, Union
_shape_t = Union[int, Tuple[int], oneflow._oneflow_internal.Size]
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)