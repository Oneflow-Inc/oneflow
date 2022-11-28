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
    oneflow.nn.Module.to_consistent,
    """
    This interface is no longer available, please use :func:`oneflow.nn.Module.to_global` instead.
    """,
)

add_docstr(
    oneflow.nn.Module.to_global,
    """
    Convert the parameters and buffers to global.

    It performs the same :func:`oneflow.Tensor.to_global` conversion to each parameter and buffer in this module.


    Note:
        This method modifies the module in-place.

        Both placement and sbp are required if the parameters and buffers of this module are local,
        otherwise at least one of placement and sbp is required.

    Args:
        placement (flow.placement, optional): the desired placement of the parameters and buffers in this module. Default: None
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp of the parameters and buffers in this module. Default: None

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> m = flow.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        >>> m.to_global(placement=flow.placement("cpu", ranks=[0]), sbp=[flow.sbp.split(0)])
        >>> m.weight.is_global
        True
        >>> m.bias.is_global
        True
    """,
)
