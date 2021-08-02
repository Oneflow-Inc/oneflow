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
    oneflow.F.cast,
    r"""
    cast(x: Tensor, *, dtype: DataType) -> Tensor

    Returns a tensor with the specified dtype.
    
    Args:
        x (Tensor): the input tensor.
        dtype (DataType): the target data type.

    For example:
    
    .. code-block:: python
        
        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = flow.F.cast(x, dtype=flow.int32)
        >>> y
        tensor([1, 2, 3], dtype=oneflow.int32)

""",
)
