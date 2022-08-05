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
    oneflow.repeat,
    """
    repeat(input, sizes) -> Tensor

    This operator repeat the input tensor to a larger size along the specified dimensions.

    Args:
        input (oneflow.Tensor): the input Tensor.
        sizes (flow.Shape or List): The number of times to repeat this tensor along each dimension.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> out = input.repeat(1, 1, 2, 2)
        >>> out.shape
        oneflow.Size([5, 3, 12, 18])
        >>> out = input.repeat(2, 1, 1, 2, 2)
        >>> out.shape
        oneflow.Size([2, 5, 3, 12, 18])
    """,
)
