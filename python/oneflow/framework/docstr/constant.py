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
    oneflow.ones_like,
    """
    Returns a tensor filled with the scalar value 1, with the same size as input.
    flow.ones_like(input) is equivalent to flow.ones(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.random.rand(5), dtype=flow.float32)
        >>> y = flow.ones_like(x)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.zeros_like,
    """
    Returns a tensor filled with the scalar value 0, with the same size as input.
    flow.zeros_like(input) is equivalent to flow.zeros(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.random.rand(5), dtype=flow.float32)
        >>> y = flow.zeros_like(x)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)

    """,
)
