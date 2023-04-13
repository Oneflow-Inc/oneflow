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
    oneflow.as_tensor,
    r"""
    as_tensor(data, dtype=None, device=None) -> Tensor
    
    Converts data into a tensor, sharing data and preserving autograd history if possible.

    If data is already a tensor with the requeseted dtype and device then data itself is returned, but if data is a tensor with a different dtype or device then it’s copied as if using data.to(dtype=dtype, device=device).

    If data is a NumPy array (an ndarray) with the same dtype and device then a tensor is constructed using oneflow.from_numpy.
    
    The interface is consistent with PyTorch.

    Args:
        data (array_like): Initial data for the tensor. Can be a list, tuple, NumPy ``ndarray``, scalar, and other types.
        dtype (oneflow.dtype, optional): the desired data type of returned tensor. Default: if ``None``, infers data type from data.
        device (oneflow.device, optional): the device of the constructed tensor. If ``None`` and data is a tensor then the device of data is used. If None and data is not a tensor then the result tensor is constructed on the CPU.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> a = np.array([1, 2, 3])
        >>> t = flow.as_tensor(a, device=flow.device('cuda'))
        >>> t
        tensor([1, 2, 3], device='cuda:0', dtype=oneflow.int64)
        >>> t[0] = -1
        >>> a
        array([1, 2, 3])

    """,
)
