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
from oneflow.framework.args_tree import ArgsTree
from oneflow.utils.global_view.global_utils import to_local_tensor


def to_local(input, *, copy=False):
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
        >>> local_state_dict = flow.utils.global_view.to_local(model.state_dict()) # doctest: +SKIP
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
        return to_local_tensor(input, copy)
    elif isinstance(input, (dict, tuple, list)):
        input_tree = ArgsTree(input)

        def leaf_fn(node):
            if isinstance(node, Tensor):
                return to_local_tensor(node, copy)
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
