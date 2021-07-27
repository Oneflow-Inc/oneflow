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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import Tensor, register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Tile(Module):
    def __init__(self, reps: tuple) -> None:
        super().__init__()
        self.reps = reps

    def forward(self, input: Tensor) -> Tensor:
        reps = self.reps
        for s in self.reps:
            assert s > 0
        input_shape = input.shape
        diff = len(input_shape) - len(reps)
        if diff > 0:
            shape = [1 for _ in range(diff)]
            shape.extend([i for i in reps])
            reps = tuple(shape)
        return input.repeat(reps)


@register_tensor_op("tile")
def tile_op(x, reps):
    """The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.tile.html

    Constructs a tensor by repeating the elements of ``input``.  The ``reps`` argument specifies the number
    of repetitions in each dimension.

    If ``reps`` specifies fewer dimensions than ``input`` has, then ones are prepended to ``reps`` until
    all dimensions are specified.  For example, if ``input`` has shape (8, 6, 4, 2) and ``reps`` is (2, 2),
    then ``reps`` is treated as (1, 1, 2, 2).

    Analogously, if ``input`` has fewer dimensions than ``reps`` specifies, then ``input`` is treated as
    if it were unsqueezed at dimension zero until it has as many dimensions as ``reps`` specifies.
    For example, if ``input`` has shape (4, 2) and ``reps`` is (3, 3, 2, 2), then ``input`` is treated as
    if it had the shape (1, 1, 4, 2).

    .. note::
        This function is similar to NumPy’s tile function.

    Args:
        input (oneflow.compatible.single_client.Tensor): the tensor whose elements to repeat.
        reps (tuple): the number of repetitions per dimension.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = np.array([1, 2]).astype(np.int32)
        >>> input = flow.Tensor(x, dtype=flow.int32)
        >>> out = input.tile(reps=(2,))
        >>> out
        tensor([1, 2, 1, 2], dtype=oneflow.int32)

        >>> x = np.random.randn(5, 2, 1)
        >>> input = flow.Tensor(x)
        >>> out = input.tile(reps=(3, 4))
        >>> out.size()
        flow.Size([5, 6, 4])

    """
    return Tile(reps=reps)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
