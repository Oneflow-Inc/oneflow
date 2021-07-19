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

import collections
from typing import Optional, Sequence, Union

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.modules.utils import _check_axis


class Flip(Module):
    def __init__(self, dims) -> None:
        super().__init__()
        assert isinstance(dims, list) or isinstance(
            dims, tuple
        ), f"dims must be list or tuple"
        self.dims = dims

    def forward(self, x):
        return flow.F.flip(x, self.dims)


@oneflow_export("flip")
@experimental_api
def floor_op(input, dims):

    r"""
    
    Reverse the order of a n-D tensor along given axis in dims.

    .. note::
        `flow.flip` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flip`,
        which returns a view in constant time. Since copying a tensor's data is more work than viewing that data,
        `flow.flip` is expected to be slower than `np.flip`.

    Args:
        input (Tensor): the input tensor
        dims (a list or tuple): axis to flip on
        
    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        
        >>> np_arr = np.arange(0, 8).reshape((2, 2, 2)).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> out = flow.flip(input, [0, 1])
        >>> out
        tensor([[[6., 7.],
                 [4., 5.]],
        <BLANKLINE>
                [[2., 3.],
                 [0., 1.]]], dtype=oneflow.float32)

    """

    return Flip(dims)(input)


@register_tensor_op("flip")
@experimental_api
def floor_op_tensor(input, dims):
    r"""
    See :func:`oneflow.experimental.flip`
    """
    return Flip(dims)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
