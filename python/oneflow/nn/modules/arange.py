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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


def arange_op(
    start: int = 0,
    end: int = None,
    step: int = 1,
    dtype: flow.dtype = None,
    device: Union[str, flow.device] = None,
    placement: flow.placement = None,
    sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    requires_grad: bool = False,
):
    if end is None:
        end = start
        start = 0
    if placement is None:
        if isinstance(device, str):
            device = flow.device(device)
        res = flow._C.arange(start, end, step, dtype=dtype, device=device)
    else:
        assert isinstance(
            placement, flow._oneflow_internal.placement
        ), "placement should be oneflow._oneflow_internal.placement type."
        assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
        if isinstance(sbp, flow.sbp.sbp):
            sbp = (sbp,)
        else:
            for elem in sbp:
                assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
        assert len(sbp) == len(placement.ranks.shape)
        res = flow._C.global_arange(
            start, end, step, dtype=dtype, placement=placement, sbp=sbp
        )

    res.requires_grad = requires_grad
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
