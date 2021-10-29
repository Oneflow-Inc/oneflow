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

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


@register_tensor_op("unfold")
def _flow_unfold(input, dimension: int, size: int, step: int):
    r"""Returns a view of the original tensor which contains all slices of `size` size from `self`
    tensor in the dimension `dimension`.

    Step between two slices is given by `step`.

    If sizedim is the size of dimension `dimension` for `self`, the size of dimension dimension in the
    returned tensor will be (sizedim - size) / step + 1.

    An additional dimension of size `size` is appended in the returned tensor.

    Args:
        dimension (int): dimension in which unfolding happens
        size (int): the size of each slice that is unfolded
        step (int): the step between each slice

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.arange(1., 8)
        >>> x
        tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> x.unfold(0, 2, 1)
        tensor([[ 1.,  2.],
                [ 2.,  3.],
                [ 3.,  4.],
                [ 4.,  5.],
                [ 5.,  6.],
                [ 6.,  7.]])
        >>> x.unfold(0, 2, 2)
        tensor([[ 1.,  2.],
                [ 3.,  4.],
                [ 5.,  6.]])
    """
    return flow._C.unfold_tensor(input, dimension, size, step)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
