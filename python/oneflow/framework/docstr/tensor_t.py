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
    oneflow.t,
    """
    oneflow.t(input) → Tensor.

        Expects `input` to be <= 2-D tensor and transposes dimensions 0 and 1. 

        0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to `transpose(input, 0, 1)`.

    Args:
        input (oneflow.Tensor): An input tensor.   
 
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.random.randn(), dtype=flow.float32)
        >>> flow.t(x).shape
        oneflow.Size([])
        >>> x = flow.tensor(np.random.randn(3), dtype=flow.float32)
        >>> flow.t(x).shape
        oneflow.Size([3])
        >>> x = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> flow.t(x).shape
        oneflow.Size([3, 2])
    
    """,
)
