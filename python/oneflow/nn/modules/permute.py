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


@register_tensor_op("permute")
def permute_op(input, *dims):
    """Returns a view of the original tensor with its dimensions permuted.

    Args:
        *dims (int...): The desired ordering of dimensions

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> out = input.permute(1, 0, 2, 3).shape
        >>> out
        flow.Size([6, 2, 5, 3])

    """

    perm = list(dims)
    assert len(perm) == len(input.shape)
    new_perm = []
    for dim in perm:
        if dim < 0:
            dim += len(perm)
        assert dim >= 0 and dim < len(
            input.shape
        ), "Invalid dim0 {}, len(shape): {}".format(dim, len(input.shape))
        new_perm.append(dim)
    return flow.F.transpose(input, perm=new_perm)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
