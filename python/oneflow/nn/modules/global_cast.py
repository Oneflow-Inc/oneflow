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
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op, Tensor
from oneflow.nn.module import Module


class ToGlobal(Module):
    def __init__(self, placement, sbp):
        super().__init__()
        self.placement = placement
        if isinstance(sbp, flow.sbp.sbp):
            sbp = [sbp]
        for elem in sbp:
            assert isinstance(
                elem, flow.sbp.sbp
            ), "element %s is not an sbp instance" % (sbp)
        self.sbp = sbp

    def forward(self, x, sbp, placement):
        return flow._C.to_global(x, placement=placement, sbp=sbp)


def to_global_op(input, placement=None, sbp=None, grad_sbp=None):
    assert isinstance(input, Tensor)

    def _check_sbp(sbp):
        if sbp is None:
            pass
        elif isinstance(sbp, (tuple, list)):
            if not all(isinstance(sbp_item, flow.sbp.sbp) for sbp_item in sbp):
                raise TypeError(
                    "sbp parameter must be type of oneflow.sbp.sbp or list/tuple of oneflow.sbp.sbp"
                )
        elif isinstance(sbp, flow.sbp.sbp):
            sbp = (sbp,)
        else:
            raise TypeError(f"Invalid parameter sbp with type {type(sbp)}")

        return sbp

    sbp = _check_sbp(sbp)

    if input.is_global:
        # convert global tensor to another global tensor with different placement or sbp
        if placement is None:
            placement = input.placement

        if sbp is None:
            sbp = input.sbp

        grad_sbp = _check_sbp(grad_sbp)

    else:
        # local tensor to global tensor
        if placement is None or sbp is None:
            raise ValueError(
                "Converting a local tensor to global tensor must have placement and sbp parameters."
            )

        if not isinstance(placement, flow.placement):
            raise ValueError(f"Invalid parameter placement with type {type(placement)}")

    if grad_sbp is None:
        grad_sbp = tuple()
    return flow._C.to_global(input, placement, sbp, grad_sbp)


class ToLocal(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow._C.to_local(x)


def to_local_op(input):
    assert input.is_global, "input must be a global tensor!"
    return flow._C.to_local(input)
