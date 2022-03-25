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
    oneflow.view_as,
    """
    view_as(x, other) -> Tensor
    This function is equivalent to torch’s view_as function.  

    View this tensor as the same size as :attr:`other`.
    ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.

    Args:
        x (Tensor): Return x's view.
        other (Tensor): The result tensor has the same size as :attr:`other`.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
               
        >>> x = flow.tensor(range(12)).reshape(3,4)
        >>> y = flow.tensor(range(12)).reshape(2,6)
        >>> flow.view_as(x, y)
        tensor([[ 0,  1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10, 11]], dtype=oneflow.int64)

    """,
)
