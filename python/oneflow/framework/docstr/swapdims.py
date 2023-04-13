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
    oneflow._C.swapdims,
    """
    swapdims(input, dim0, dim1) -> Tensor

    This function is equivalent to torch’s swapdims function.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow

        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> x
        tensor([[[0, 1],
                 [2, 3]],
        <BLANKLINE>
                [[4, 5],
                 [6, 7]]], dtype=oneflow.int64)
        >>> flow.swapdims(x, 0, 1)
        tensor([[[0, 1],
                 [4, 5]],
        <BLANKLINE>
                [[2, 3],
                 [6, 7]]], dtype=oneflow.int64)
        >>> flow.swapdims(x, 0, 2)
        tensor([[[0, 4],
                 [2, 6]],
        <BLANKLINE>
                [[1, 5],
                 [3, 7]]], dtype=oneflow.int64)

    """,
)
