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
    pin_memory: bool = False,
):
    assert size is not None, "shape must not be None"

    shape = _single(_handle_size_arg(size))

    if dtype is None:
        dtype = flow.get_default_dtype()
    if placement is not None:
        assert (
            device is None
        ), "argument 'device' must be None when argument 'placement' exist"

    if placement is not None:
        assert (
            sbp is not None
        ), "argument 'sbp' must not be None when argument 'placement' exist"
        assert isinstance(
            sbp, (flow.sbp.sbp, tuple, list)
        ), f"argument 'sbp' must be flow.sbp.sbp, not %s" % (type(sbp))
        if isinstance(sbp, flow.sbp.sbp):
            sbp = (sbp,)
        else:
            for elem in sbp:
                assert isinstance(elem, flow.sbp.sbp), (
                    "Element in argument 'sbp' must be flow.sbp.sbp, not %s"
                    % (type(elem))
                )
        assert len(sbp) == len(placement.ranks.shape)
    else:
        assert sbp is None, "argument 'sbp' must be None"

    if placement is not None:
        tensor = flow._C.global_empty(shape, dtype=dtype, placement=placement, sbp=sbp)
        tensor.requires_grad_(requires_grad)
    else:
        tensor = flow._C.empty(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    return tensor


def empty_like_op(
    input,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False,
):
    new_size = _single(_handle_size_arg(input.size()))
    if placement is None and input.is_global and input.placement is not None:
        placement = input.placement
    if sbp is None and input.is_global and input.sbp is not None:
        sbp = input.sbp
    if dtype is None:
        dtype = input.dtype
    if placement is None and device is None:
        device = input.device
    return empty_op(
        new_size,
        dtype=dtype,
        device=device,
        placement=placement,
        sbp=sbp,
        requires_grad=requires_grad,
    )


def new_empty_op(
    x, size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False
):
    new_size = _single(_handle_size_arg(size))
    new_dtype = dtype
    new_device = device
    new_placement = placement
    new_sbp = sbp

    if dtype is None:
        new_dtype = x.dtype
    if device is None:
        new_device = x.device if x.is_local else None
    if placement is None:
        new_placement = x.placement if x.is_global else None
    if sbp is None:
        new_sbp = x.sbp if x.is_global else None

    return empty_op(
        new_size,
        dtype=new_dtype,
        device=new_device,
        placement=new_placement,
        sbp=new_sbp,
        requires_grad=requires_grad,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
