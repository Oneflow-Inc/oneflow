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
from oneflow.nn.modules.module import Module


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


def local_to_global_op(input, placement=None, sbp=None, *, check_meta=True, copy=False):
    # Convert None to a tensor with shape 0, in order to input it into flow._C.to_global.
    if input is None:
        input = flow.tensor(())

    assert isinstance(input, Tensor)
    assert input.is_local, "input must be a local tensor"
    if placement is None or sbp is None:
        raise ValueError(
            "Converting a local tensor to global tensor must have placement and sbp parameters."
        )

    assert isinstance(
        placement, flow.placement
    ), f"Invalid parameter placement with type {type(placement)}"

    sbp = _check_sbp(sbp)
    grad_sbp = tuple()
    return flow._C.to_global(input, placement, sbp, grad_sbp, check_meta, copy)


def global_to_global_op(
    input, placement=None, sbp=None, *, grad_sbp=None, check_meta=False, copy=False
):
    assert isinstance(input, Tensor)
    assert input.is_global, "input must be a global tensor"

    sbp = _check_sbp(sbp)
    if placement is None:
        placement = input.placement

    if sbp is None:
        sbp = input.sbp

    assert isinstance(
        placement, flow.placement
    ), f"Invalid parameter placement with type {type(placement)}"

    grad_sbp = _check_sbp(grad_sbp)
    if grad_sbp is None:
        grad_sbp = tuple()
    return flow._C.to_global(input, placement, sbp, grad_sbp, check_meta, copy)


def to_global_op(input, placement=None, sbp=None, **kwargs):
    assert isinstance(input, Tensor)

    if input.is_global:
        return global_to_global_op(input=input, placement=placement, sbp=sbp, **kwargs)
    else:
        if "grad_sbp" in kwargs:
            del kwargs["grad_sbp"]
        return local_to_global_op(input=input, placement=placement, sbp=sbp, **kwargs)


def to_local_op(input, *, copy=False):
    assert input.is_global, "Expected global tensor for to_local but got local tensor!"
    return flow._C.to_local(input, copy)
