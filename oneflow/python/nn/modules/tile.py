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
from typing import Union
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import Tensor, register_tensor_op


class Tile(Module):
    def __init__(self, sizes: Union[tuple, flow.Size]) -> None:
        super().__init__()
        self.sizes = sizes

    def forward(self, input: Tensor) -> Tensor:
        sizes = self.sizes
        for s in self.sizes:
            assert s > 0
        input_shape = input.shape
        diff = len(input_shape) - len(sizes)
        if diff > 0:
            shape = [1 for _ in range(diff)]
            shape.extend([i for i in sizes])
            sizes = tuple(shape)
        return input.repeat(sizes)


@oneflow_export("tile")
@register_tensor_op("tile")
@experimental_api
def tile_op(x, sizes):
    r"""The interface is consistent with PyTorch.

    This operator repeat the input tensor to a larger size along the specified dimensions.

    Args:
        x (oneflow.Tensor): The input Tensor.
        size (Sequence[int]): The number of times to repeat this tensor along each dimension

    .. note::
        This function is similar to NumPy’s tile function.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = np.array([1, 2]).astype(np.int32)
        >>> input = flow.Tensor(x, dtype=flow.int32)
        >>> out = input.tile(sizes=(2,))
        >>> out
        tensor([1, 2, 1, 2], dtype=oneflow.int32)

        >>> x = np.random.randn(5, 2, 1)
        >>> input = flow.Tensor(x)
        >>> out = input.tile(sizes=(3, 4))
        >>> out.size()
        flow.Size([5, 6, 4])

    """
    return Tile(sizes=sizes)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
