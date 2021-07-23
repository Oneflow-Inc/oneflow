from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import Tensor
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.init import (
    _calculate_fan_in_and_fan_out,
)
from typing import Optional, List, Tuple
import math

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
