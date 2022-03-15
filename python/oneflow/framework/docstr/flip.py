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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.flip,
    """
    flip(input, dims) -> Tensor

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

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> np_arr = np.arange(0, 8).reshape((2, 2, 2)).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> input.shape
        oneflow.Size([2, 2, 2])
        >>> out = flow.flip(input, [0, 1])
        >>> out
        tensor([[[6., 7.],
                 [4., 5.]],
        <BLANKLINE>
                [[2., 3.],
                 [0., 1.]]], dtype=oneflow.float32)

    """,
)
