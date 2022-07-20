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
from typing import List, Union
import oneflow as flow


def arange_op(
    start: Union[int, flow.Tensor] = None,
    end: Union[int, flow.Tensor] = None,
    step: int = 1,
    dtype: flow.dtype = None,
    device: Union[str, flow.device] = None,
    placement: flow.placement = None,
    sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    requires_grad: bool = False,
):
    if start is None:
        start = 0
    elif flow.is_tensor(start):
        # support start as a Scalar Tensor
        assert len(start.shape) == 0, "start must be a Scalar"
        start = start.item()

    if end is None:
        end = start
        start = 0
    elif flow.is_tensor(end):
        # support end as a Scalar Tensor
        assert len(end.shape) == 0, "end must be a Scalar"
        end = end.item()

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
