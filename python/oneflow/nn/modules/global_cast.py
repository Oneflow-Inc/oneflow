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


def _to_global_tensor(input_tensor, placement=None, sbp=None, **kwargs):
    if input_tensor.is_global:
        return global_to_global_op(input=input_tensor, placement=placement, sbp=sbp, **kwargs)
    else:
        if "grad_sbp" in kwargs:
            del kwargs["grad_sbp"]
        return local_to_global_op(input=input_tensor, placement=placement, sbp=sbp, **kwargs)


def to_global_op(input, placement=None, sbp=None, **kwargs):
    r"""Converts the input tensor or input tensor(s) in list/tuple/dict to global tensor(s).
    
    Note:
        Both placement and sbp are required if the input is local, otherwise at least one of placement and sbp is required.

    Args:
        input (flow.Tensor/None/list/tuple/dict): the input that needs to be converted.
        placement (flow.placement, optional):  the desired placement of the input. Default: None
        sbp (flow.sbp.sbp or list/tuple of flow.sbp.sbp, optional): the desired sbp of the input. Default: None
    
    Returns:
        The converted input.

    For a tensor input: please refer to the examples in :func:`oneflow.Tensor.to_global`.

    For an input of other type (take a state dict as an example):

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> from oneflow import nn
        >>> placement = flow.placement("cpu", ranks=[0, 1]) # doctest: +SKIP
        >>> sbp = (flow.sbp.broadcast,) # doctest: +SKIP
        >>> model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)) # doctest: +SKIP
        >>> global_state_dict = flow.to_global(model.state_dict(), placement, sbp) # doctest: +SKIP
        >>> for val in state_dict.values(): # doctest: +SKIP
        >>>     print(val.is_global) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        True
        True
        True
        True

    .. code-block:: python

        >>> # results on rank 1
        True
        True
        True
        True
    """
    if isinstance(input, Tensor):
        return _to_global_tensor(input, placement, sbp, **kwargs)
    else:
        input_tree = IONode(value=input)

        def leaf_fn(node):
            if isinstance(node._value, Tensor):
                return _to_global_tensor(node._value, placement, sbp, **kwargs)
            else:
                warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(node._value)))
                return node._value

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input


def _to_local_tensor(input_tensor):
    if not input_tensor.is_global:
        warnings.warn("The tensor should be global, local tensor will remain the same.")
        return input_tensor
    return flow._C.to_local(input_tensor)


def to_local_op(input):
    r"""Returns the local part of the input.
    
    Returns:
        The converted input.

    For a tensor input: please refer to the examples in :func:`oneflow.Tensor.to_local`.

    For an input of other type (take a state dict as an example):

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> from oneflow import nn
        >>> placement = flow.placement("cpu", ranks=[0, 1]) # doctest: +SKIP
        >>> sbp = (flow.sbp.broadcast,) # doctest: +SKIP
        >>> model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)) # doctest: +SKIP
        >>> model = model.to_global(placement=placement, sbp=sbp) # doctest: +SKIP
        >>> local_state_dict = flow.to_local(model.state_dict()) # doctest: +SKIP
        >>> for val in local_state_dict.values(): # doctest: +SKIP
        >>>     print(val.is_global) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        False
        False
        False
        False

    .. code-block:: python

        >>> # results on rank 1
        False
        False
        False
        False
    """
    if isinstance(input, Tensor):
        return _to_local_tensor(input)
    else:
        input_tree = IONode(value=input)

        def leaf_fn(node):
            if isinstance(node._value, Tensor):
                return _to_local_tensor(node._value)
            else:
                warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(node._value)))
                return node._value

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input
