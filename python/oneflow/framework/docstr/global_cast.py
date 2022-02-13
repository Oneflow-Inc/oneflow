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
    oneflow.to_global,
    """
    to_global(input, placement=None, sbp=None, grad_sbp=None) -> Tensor

    Cast a local tensor to global tensor or cast a
    global tensor to another global tensor with
    different sbp or placement


    Args:
        input (Tensor): the input tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the input tensor must be consistent one and use its own placement.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the input tensor must be consistent one and use its own sbp.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> output_tensor = input.to_global(placement, [flow.sbp.split(0)])
        >>> output_tensor.is_global
        True
    """,
)

add_docstr(
    oneflow.to_local,
    """
    to_local(input) -> Tensor

    Returns the local tensor of a global tensor.


    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.tensor(np_arr, dtype=flow.float32)
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> global_tensor = input.to_global(placement, [flow.sbp.split(0)])
        >>> global_tensor.to_local()
        tensor([0.5000, 0.6000, 0.7000], dtype=oneflow.float32)
    """,
)
