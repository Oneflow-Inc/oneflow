from typing import Optional

import numpy as np

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Argwhere(Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        if dtype == None:
            dtype = flow.int32
        self.dtype = dtype

    def forward(self, x):
        (res, size) = flow.F.argwhere(x, dtype=self.dtype)
        slice_tup_list = [[0, int(size.numpy()), 1]]
        return flow.experimental.slice(res, slice_tup_list=slice_tup_list)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
