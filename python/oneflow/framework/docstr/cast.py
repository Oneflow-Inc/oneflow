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
    oneflow.cast,
    """
    
    The operation takes input tensor `x` and casts it to the output with `dtype`

    Args:
        x (oneflow.Tensor): A Tensor
        dtype (flow.dtype): Data type of the output tensor

    Returns:
        oneflow.Tensor: A Tensor with specific dtype.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input = flow.tensor(np_arr, dtype=flow.float32)
        >>> output = flow.cast(input, flow.int8)
        >>> np.array_equal(output.numpy(), np_arr.astype(np.int8))
        True

    """,
)
