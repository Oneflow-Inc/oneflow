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
    oneflow.flatten,
    """Flattens a contiguous range of dims into a tensor.

    Args:
        start_dim: first dim to flatten (default = 0).
        end_dim: last dim to flatten (default = -1).
    
    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.randn(32, 1, 5, 5)
        >>> output = flow.flatten(input, start_dim=1)
        >>> output.shape
        oneflow.Size([32, 25])

    """,
)
