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
from typing import Sequence

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


def _input_args_is_flow_size(args):
    return all((isinstance(x, flow.Size) for x in args)) and len(args) == 1


def reshape_op(input, shape: Sequence[int] = None):
    """This operator reshapes a Tensor.

    We can set one dimension in `shape` as `-1`, the operator will infer the complete shape.

    Args:
        x: A Tensor.
        shape: Shape of the output tensor.
    Returns:
        A Tensor has the same type as `x`.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array(
        ...    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        ... ).astype(np.float32)
        >>> input = flow.Tensor(x)

        >>> y = flow.reshape(input, shape=[2, 2, 2, -1]).shape
        >>> y
        oneflow.Size([2, 2, 2, 2])

    """
    return flow._C.reshape(input, shape)


def view_op(input, *shape):
    if len(shape) == 1:
        new_shape = shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
    else:
        new_shape = shape
    return flow._C.view(input, new_shape)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
