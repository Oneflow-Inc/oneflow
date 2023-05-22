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
from oneflow.framework.args_tree import ArgsTree


def to_global_tensor(input_tensor, placement=None, sbp=None, **kwargs):
    # specific operation for None
    if input_tensor is None:
        return flow.local_to_global(
            input=input_tensor, placement=placement, sbp=sbp, **kwargs
        )

    if input_tensor.is_global:
        return flow.global_to_global(
            input=input_tensor, placement=placement, sbp=sbp, **kwargs
        )
    else:
        if "grad_sbp" in kwargs:
            del kwargs["grad_sbp"]
        return flow.local_to_global(
            input=input_tensor, placement=placement, sbp=sbp, **kwargs
        )


def to_local_tensor(input_tensor, copy):
    if not input_tensor.is_global:
        warnings.warn("The tensor should be global, local tensor will remain the same.")
        return input_tensor
    return flow._C.to_local(input_tensor, copy)


def check_input_global(input):
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
                        assert (
                            arg.is_global == is_input_global
                        ), "Tensor(s) in the input must be all local or all global."

    return is_input_global


def check_placement_on_all_ranks(placement):
    # Determine whether the ranks of placement are same as all ranks
    is_placement_on_all_ranks = False
    all_ranks = flow.placement.all("cpu").ranks
    if (
        all_ranks.shape == placement.ranks.shape
        and (all_ranks == placement.ranks).all()
    ):
        is_placement_on_all_ranks = True

    return is_placement_on_all_ranks


def src_sbp_broadcast(obj, src: int = 0):
    rank = flow.env.get_rank()
    if src == rank:
        obj_bytes = pickle.dumps(obj)
        obj_bytes = flow._oneflow_internal.cpu_broadcast(obj_bytes, src)
    else:
        obj_bytes = flow._oneflow_internal.cpu_broadcast(None, src)
    return pickle.loads(obj_bytes)
