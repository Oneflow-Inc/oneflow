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
from typing import Optional, Sequence

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.utils import _single


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


def permute_op(input, dims):
    """Returns a view of the original tensor with its dimensions permuted.

    Args:
        dims (tuple of python:ints): The desired ordering of dimensions

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> out = flow.permute(input, (1, 0, 2, 3)).shape
        >>> out
        oneflow.Size([6, 2, 5, 3])

    """
    if _input_args_is_int(dims):
        new_dims = _single(dims)
    else:
        raise ValueError("the input dims parameter of permute is not illegal!")
    return flow._C.transpose(input, perm=new_dims)


@register_tensor_op("permute")
def permute_tensor_op(input, *dims):
    """Returns a view of the original tensor with its dimensions permuted.

    Args:
        *dims (int...): The desired ordering of dimensions

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> out = input.permute(1, 0, 2, 3).shape
        >>> out
        oneflow.Size([6, 2, 5, 3])

    """

    if _input_args_is_int(dims):
        new_dims = _single(dims)
    elif _input_args_is_int(*dims):
        new_dims = _single(*dims)
    else:
        raise ValueError("the input dims parameter of permute is not illegal!")
    return flow._C.transpose(input, perm=new_dims)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
