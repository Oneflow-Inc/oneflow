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
import warnings

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn import Parameter
from oneflow.nn.module import Module
from oneflow.nn.graph.util import IONode


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


def local_to_global_op(input, placement=None, sbp=None, *, check_meta=True):
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
    return flow._C.to_global(input, placement, sbp, grad_sbp, check_meta)


def global_to_global_op(
    input, placement=None, sbp=None, *, grad_sbp=None, check_meta=False
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
    return flow._C.to_global(input, placement, sbp, grad_sbp, check_meta)


def to_global_op(input, placement=None, sbp=None, **kwargs):
    assert isinstance(input, Tensor)

    if input.is_global:
        return global_to_global_op(input=input, placement=placement, sbp=sbp, **kwargs)
    else:
        if "grad_sbp" in kwargs:
            del kwargs["grad_sbp"]
        return local_to_global_op(input=input, placement=placement, sbp=sbp, **kwargs)


def to_global_op(input, placement=None, sbp=None, **kwargs):
    r"""Converts to a global tensor or state dict if it is local, otherwise performs designated placement and/or sbp conversion.
    
    It performs the same conversion as :func:`oneflow.Tensor.to_global` to the tensor or every tensor in the state dict.

    Note:
        Both placement and sbp are required if the input is local, otherwise at least one of placement and sbp is required.

    Args:
        input (flow.Tensor or dict):  
        placement (flow.placement, optional):  the desired placement of the parameters and buffers in this module. Default: None
        sbp (flow.sbp.sbp or list/tuple of flow.sbp.sbp, optional): the desired sbp of the parameters and buffers in this module. Default: None
    
    Returns:
        The converted tensor / state dict.
    """
    assert isinstance(input, (Tensor, dict)), "input must be a tensor/dict!"

    sbp = _check_sbp(sbp)

    # for a tensor input
    if isinstance(input, Tensor): 
        if input.is_global:
            return global_to_global_op(input=input, placement=placement, sbp=sbp, **kwargs)
        else:
            if "grad_sbp" in kwargs:
                del kwargs["grad_sbp"]
            return local_to_global_op(input=input, placement=placement, sbp=sbp, **kwargs)

    # for a state dict input
    elif isinstance(input, dict):
        input_tree = IONode(value=input)

        def leaf_fn(node):
            if isinstance(node._value, Tensor):
                if node._value.is_global:
                    return global_to_global_op(input=node._value, placement=placement, sbp=sbp, **kwargs)
                else:
                    if "grad_sbp" in kwargs:
                        del kwargs["grad_sbp"]
                    return local_to_global_op(input=node._value, placement=placement, sbp=sbp, **kwargs)
            return node._value

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input


def to_local_op(input):
    r"""Returns the local component of the input.

    It performs the same conversion as :func:`oneflow.Tensor.to_local` to the tensor(s) in the input.
    
    Returns:
        The converted input.
    """
    assert isinstance(input, (Tensor, dict)), "input must be a tensor/dict!"

    # for a tensor input
    if isinstance(input, Tensor):
        assert input.is_global, "input tensor must be global!"
        return flow._C.to_local(input)

    # for a state dict input
    elif isinstance(input, dict):
        input_tree = IONode(value=input)

        def leaf_fn(node):
            if isinstance(node._value, Tensor):
                assert node._value.is_global, "the tensor must be global!"
                return Parameter(node._value.to_local())
                
            return node._value

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input
