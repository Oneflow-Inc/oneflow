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

from typing import List, Optional, Union

import oneflow as flow
from oneflow.nn.common_types import _size_any_t
from oneflow.nn.modules.utils import _handle_size_arg, _single


def empty_op(
    *size,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp]
    ] = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with uninitialized data.
    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        device (torch.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp or List[flow.sbp], optional): The desired sbp of returned global tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.empty(4, 5)  # construct local empty tensor
        >>> y.shape
        oneflow.Size([4, 5])
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.empty(4, 5, placement=placement, sbp=flow.sbp.broadcast)  # construct consistent empty tensor
        >>> y.is_global
        True

    """
    assert size is not None, "shape must not be None"

    shape = _single(_handle_size_arg(size))
    if dtype is None:
        dtype = flow.float32
    if placement is None:
        if device is None:
            device = flow.device("cpu")
    else:
        assert device is None

    if placement is not None:
        assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
        if isinstance(sbp, flow.sbp.sbp):
            sbp = (sbp,)
        else:
            for elem in sbp:
                assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
        assert len(sbp) == len(placement.ranks.shape)
    else:
        assert sbp is None, "sbp: %s" % sbp

    if placement is not None:
        tensor = flow._C.global_empty(shape, dtype=dtype, placement=placement, sbp=sbp)
    else:
        tensor = flow._C.empty(shape, dtype=dtype, device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
