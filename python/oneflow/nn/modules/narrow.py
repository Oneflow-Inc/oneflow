"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

import oneflow as flow
from oneflow.framework.tensor import Tensor, register_tensor_op
from oneflow.nn.module import Module


class Narrow(Module):
    def __init__(self, dim: int, start: int, length: int) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x):
        dim = dim + x.dim if self.dim < 0 else self.dim
        return flow._C.narrow(x, dim=dim, start=self.start, length=self.length)


@register_tensor_op("narrow")
def narrow_op(x, dim: int, start: int, length: int):
    """Returns a new tensor that is a narrowed version of `x` tensor.
    The dimension `dim` is input from `start` to `start + length`.

    Args:
        x: the tensor to narrow.
        dim: the dimension along which to narrow.
        start: the starting dimension.
        length: the distance to the ending dimension.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> flow.narrow(x, 0, 0, 2)
        tensor([[1, 2, 3],
                [4, 5, 6]], dtype=oneflow.int64)
        >>> flow.narrow(x, 1, 1, 2)
        tensor([[2, 3],
                [5, 6],
                [8, 9]], dtype=oneflow.int64)
    """
    return Narrow(dim, start, length)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
