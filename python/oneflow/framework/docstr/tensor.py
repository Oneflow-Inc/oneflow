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
    oneflow.tensor,
    r"""
    Constructs a tensor with data, return a consistent tensor if placement and sbp are in kwargs,
       otherwise return a local tensor. 
       
    Arguments:
        data: Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar or tensor.
    Keyword Arguments:
        dtype (oneflow.dtype, optional) – the desired data type of returned tensor.
            Default: if None, infers data type from data.
        device (oneflow.device, optional): the desired device of returned tensor. If placement
            and sbp is None, uses the current cpu for the default tensor type.
        placement (oneflow.placement, optional): the desired placement of returned tensor.
        sbp (oneflow.sbp or tuple of oneflow.sbp, optional): the desired sbp of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False

    Noted:
        The Keyword Argument device is mutually exclusive with placement and sbp.
        Consistent tensor only can be constructed from tensor.


    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
)
