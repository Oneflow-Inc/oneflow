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
import pickle

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.graph.util import ArgsTree


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
    # convert None to a tensor with shape 0, in order to input it into flow._C.to_global
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
    # specific operation for None
    if input_tensor is None:
        return local_to_global_op(input=input_tensor, placement=placement, sbp=sbp, **kwargs)
    
    if input_tensor.is_global:
        return global_to_global_op(input=input_tensor, placement=placement, sbp=sbp, **kwargs)
    else:
        if "grad_sbp" in kwargs:
            del kwargs["grad_sbp"]
        return local_to_global_op(input=input_tensor, placement=placement, sbp=sbp, **kwargs)


def _check_input_global(input):
    is_input_global = False
    if input is not None:
        if isinstance(input, Tensor):
            is_input_global = input.is_global
        elif isinstance(input, (dict, tuple, list)):
            is_first_tensor_in_input = True
            input_tree_for_is_global = ArgsTree(input)
            for arg in input_tree_for_is_global.iter_nodes():
                if isinstance(arg, Tensor):
                    if is_first_tensor_in_input:
                        is_input_global = arg.is_global
                        is_first_tensor_in_input = False
                    else:
                        assert arg.is_global == is_input_global, "Tensor(s) in the input must be all local or all global."

    return is_input_global


def _check_placement_on_all_ranks(placement):
    """Determine whether the ranks of placement are same as all ranks
    """
    is_placement_on_all_ranks = False
    all_ranks = flow.env.all_device_placement("cpu").ranks
    if all_ranks.shape == placement.ranks.shape and (all_ranks == placement.ranks).all():
        is_placement_on_all_ranks = True

    return is_placement_on_all_ranks


def to_global_op(input, placement=None, sbp=None, **kwargs):
    r"""Converts the input tensor or input tensor(s) in list/tuple/dict to global tensor(s).
    
    Note:
        Both placement and sbp are required if the input is local, otherwise at least one of placement and sbp is required.

    Args:
        input (oneflow.Tensor/None/list/tuple/dict): the input that needs to be converted.
        placement (oneflow.placement, optional):  the desired placement of the input. Default: None
        sbp (oneflow.sbp.sbp or list/tuple of oneflow.sbp.sbp, optional): the desired sbp of the input. Default: None
    
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
    is_input_not_tensor_or_none = False
    if (input is not None) and (not isinstance(input, (Tensor, dict, tuple, list))):
        is_input_not_tensor_or_none = True

    if (not is_input_not_tensor_or_none) and (placement is not None) and (not _check_input_global(input)) and \
        (not _check_placement_on_all_ranks(placement)):
        src_rank = placement.ranks.flat[0]
        cur_rank = flow.env.get_rank()

        if cur_rank == src_rank:
            # Replace tensor(s) in the input with None, in order to reduce communication cost
            if isinstance(input, Tensor) or input is None:
                mapped_input_none = None
            else:
                input_tree_none = ArgsTree(input)

                def leaf_fn_to_none(node):
                    if isinstance(node, Tensor):
                        return None
                    else:
                        warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(node)))
                        return node

                mapped_input_none = input_tree_none.map_leaf(leaf_fn_to_none)

            obj_input = pickle.dumps(mapped_input_none)
            flow._oneflow_internal.cpu_broadcast(obj_input, src_rank)
        else:
            if cur_rank in placement.ranks:
                # Participating in the broadcast process but retaining original value
                flow._oneflow_internal.cpu_broadcast(None, src_rank)
            else:
                # The input of other ranks will be always overwritten no matter what is passed in
                input = pickle.loads(flow._oneflow_internal.cpu_broadcast(None, src_rank))

    if isinstance(input, Tensor) or input is None:
        return _to_global_tensor(input, placement, sbp, **kwargs)
    elif isinstance(input, (dict, tuple, list)):
        input_tree = ArgsTree(input)

        def leaf_fn(node):
            if isinstance(node, Tensor) or node is None:
                return _to_global_tensor(node, placement, sbp, **kwargs)
            else:
                warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(node)))
                return node

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input
    else:
        warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(input)))
        return input


def _flatten_dict(input_dict, output_dict=None, parent_key=None, separator="."):
    """ Flatten any nested dict.
    """
    if output_dict is None:
        output_dict = {}
    
    for key, value in input_dict.items():
        full_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            _flatten_dict(value, output_dict=output_dict, parent_key=full_key)
        else:
            output_dict[full_key] = value

    return output_dict


def _restore_flat_dict(flat_dict, original_dict, output_dict=None, parent_key="", separator="."):
    """ Restore a flat dict according to the original dict's structure, reverse operation of _flatten_dict function.
    """
    if output_dict is None:
        output_dict = {}
    
    for key, value in original_dict.items():
        full_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            output_dict[key] = {}
            _restore_flat_dict(flat_dict, value, output_dict=output_dict[key], parent_key=full_key,
                               separator=separator)
        else:
            output_dict[key] = flat_dict[full_key]
            
    return output_dict


def dict_to_global(input_dict, placement, sbp, *, sbp_for_special_keys):
    r"""Converts the input dict containing tensor(s) to global one, supporting to assign different SBP(s) for special
        key(s). Usually used for making graph models's state dict global.

    Args:
        input_dict (dict): the input dict that needs to be converted.
        placement (oneflow.placement): the common placement for all keys.
        sbp (oneflow.sbp.sbp): the default SBP for not special keys.
        sbp_for_special_keys (dict): The keys are str type, and the values are oneflow.sbp.sbp type.
         This is used to specify special SBPs for "System-Train-TrainStep" tensor with shape [1] and 
         other insplitable "small tensors". The keys here support the use of "." to indicate nested dicts.
    """
    assert isinstance(input_dict, dict) or input_dict is None, "The input_dict must be a dict or None!"
    
    if (not _check_input_global(input_dict)) and (not _check_placement_on_all_ranks(placement)):
        src_rank = placement.ranks.flat[0]
        cur_rank = flow.env.get_rank()

        if cur_rank == src_rank:
            # Replace tensor(s) in the input with None, in order to reduce communication cost
            if input_dict is None:
                mapped_input_none = None
            else:
                input_tree_none = ArgsTree(input_dict)

                def leaf_fn_to_none(node):
                    if isinstance(node, Tensor):
                        return None
                    else:
                        warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(node)))
                        return node

                mapped_input_none = input_tree_none.map_leaf(leaf_fn_to_none)

            obj_input = pickle.dumps(mapped_input_none)
            flow._oneflow_internal.cpu_broadcast(obj_input, src_rank)
        else:
            if cur_rank in placement.ranks:
                # Participating in the broadcast process but retaining original value
                flow._oneflow_internal.cpu_broadcast(None, src_rank)
            else:
                # The input of other ranks will be always overwritten no matter what is passed in
                input_dict = pickle.loads(flow._oneflow_internal.cpu_broadcast(None, src_rank))

    flat_dict = _flatten_dict(input_dict)

    for key in flat_dict.keys():
        if key not in sbp_for_special_keys.keys():
            flat_dict[key] = _to_global_tensor(flat_dict[key], placement=placement, sbp=sbp)
        else:
            flat_dict[key] = _to_global_tensor(flat_dict[key], placement=placement, sbp=sbp_for_special_keys[key])

    input_dict = _restore_flat_dict(flat_dict, input_dict)
    return input_dict


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
    elif isinstance(input, (dict, tuple, list)):
        input_tree = ArgsTree(input)

        def leaf_fn(node):
            if isinstance(node, Tensor):
                return _to_local_tensor(node)
            else:
                warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(node)))
                return node

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input
    else:
        warnings.warn("Non-Tensor type: {} encountered, it will remain the same.".format(type(input)))
        return input
