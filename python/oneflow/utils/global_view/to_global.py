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


def to_global(input, placement=None, sbp=None, **kwargs):
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

    if (
        (not is_input_not_tensor_or_none)
        and (placement is not None)
        and (not check_input_global(input))
        and (not check_placement_on_all_ranks(placement))
    ):
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
                input = pickle.loads(
                    flow._oneflow_internal.cpu_broadcast(None, src_rank)
                )

    if isinstance(input, Tensor) or input is None:
        return to_global_tensor(input, placement, sbp, **kwargs)
    elif isinstance(input, (dict, tuple, list)):
        input_tree = ArgsTree(input)

        def leaf_fn(node):
            if isinstance(node, Tensor) or node is None:
                return to_global_tensor(node, placement, sbp, **kwargs)
            else:
                warnings.warn(
                    "Non-Tensor type: {} encountered, it will remain the same.".format(
                        type(node)
                    )
                )
                return node

        mapped_input = input_tree.map_leaf(leaf_fn)
        return mapped_input
    else:
        warnings.warn(
            "Non-Tensor type: {} encountered, it will remain the same.".format(
                type(input)
            )
        )
        return input
