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
from oneflow.framework.tensor import register_tensor_op


@register_tensor_op("tile")
def tile_op(input, reps):
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
        input (oneflow.Tensor): the tensor whose elements to repeat.
        reps (tuple): the number of repetitions per dimension.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = np.array([1, 2]).astype(np.int32)
        >>> input = flow.tensor(x, dtype=flow.int32)
        >>> out = input.tile(reps=(2,))
        >>> out
        tensor([1, 2, 1, 2], dtype=oneflow.int32)

        >>> x = np.random.randn(5, 2, 1)
        >>> input = flow.Tensor(x)
        >>> out = input.tile(reps=(3, 4))
        >>> out.size()
        oneflow.Size([5, 6, 4])

    """

    for s in reps:
        assert s > 0
    input_shape = input.shape
    diff = len(input_shape) - len(reps)
    if diff > 0:
        shape = [1 for _ in range(diff)]
        shape.extend([i for i in reps])
        reps = tuple(shape)
    return input.repeat(reps)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
