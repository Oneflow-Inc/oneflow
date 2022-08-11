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
from oneflow.utils.global_view.global_utils import (
    to_global_tensor,
    check_input_global,
    check_placement_on_all_ranks,
)


def dict_to_global(input_dict, placement, sbp, *, sbp_for_special_keys):
    r"""Converts the input dict containing tensor(s) to global one, supporting to assign different SBP(s) for special key(s). Usually used for making graph models's state dict global.

    Args:
        input_dict (dict): the input dict that needs to be converted.
        placement (oneflow.placement): the common placement for all keys.
        sbp (oneflow.sbp.sbp): the default SBP for not special keys.
        sbp_for_special_keys (dict): The keys are str type, and the values are oneflow.sbp.sbp type.
         This is used to specify special SBPs for "System-Train-TrainStep" tensor with shape [1] etc.
         Also, it is worth noting that, for a tensor of shape `(1, n)`, you can specify SBP is `oneflow.sbp.split(1)`.
         The keys here support the use of "." to indicate nested dicts.

    Note:
        For example, there is a state dict with `{"module_pipeline": {"m_stage0.linear.weight": ..., ...}, ...}`, and you want to specify the SBP for `m_stage0.linear.weight` as `oneflow.sbp.split(1)`,
        the above sbp_for_special_keys (dict) parameter can be assigned `{"module_pipeline.m_stage1.linear.weight": oneflow.sbp.split(1)}`.
    """
    assert (
        isinstance(input_dict, dict) or input_dict is None
    ), "The input_dict must be a dict or None!"

    if (not check_input_global(input_dict)) and (
        not check_placement_on_all_ranks(placement)
    ):
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
                        warnings.warn(
                            "Non-Tensor type: {} encountered, it will remain the same.".format(
                                type(node)
                            )
                        )
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
                input_dict = pickle.loads(
                    flow._oneflow_internal.cpu_broadcast(None, src_rank)
                )

    flat_dict = _flatten_dict(input_dict)

    for key in flat_dict.keys():
        if key not in sbp_for_special_keys.keys():
            flat_dict[key] = to_global_tensor(
                flat_dict[key], placement=placement, sbp=sbp
            )
        else:
            flat_dict[key] = to_global_tensor(
                flat_dict[key], placement=placement, sbp=sbp_for_special_keys[key]
            )

    input_dict = _restore_flat_dict(flat_dict, input_dict)
    return input_dict


def _flatten_dict(input_dict, output_dict=None, parent_key=None, separator="."):
    # Flatten any nested dict.
    if output_dict is None:
        output_dict = {}

    for key, value in input_dict.items():
        full_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            _flatten_dict(value, output_dict=output_dict, parent_key=full_key)
        else:
            output_dict[full_key] = value

    return output_dict


def _restore_flat_dict(
    flat_dict, original_dict, output_dict=None, parent_key="", separator="."
):
    # Restore a flat dict according to the original dict's structure, reverse operation of _flatten_dict function.
    if output_dict is None:
        output_dict = {}

    for key, value in original_dict.items():
        full_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            output_dict[key] = {}
            _restore_flat_dict(
                flat_dict,
                value,
                output_dict=output_dict[key],
                parent_key=full_key,
                separator=separator,
            )
        else:
            output_dict[key] = flat_dict[full_key]

    return output_dict
