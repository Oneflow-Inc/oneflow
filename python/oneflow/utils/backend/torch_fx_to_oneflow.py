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
import re
import oneflow
import torch
from torch import fx

mapping_dict = {}

internal_oneflow_funcs = [
    "FunctionConfig",
    "Generator",
    "INVALID_SPLIT_AXIS",
    "MultiClientSession",
    "Tensor",
    "builtin_op",
    "distributed",
    "default_generator",
    "docstr",
    "eager",
    "enable_eager_execution",
    "env",
    "framework",
]


oneflow_funcs = dir(oneflow)
for funcs_name in oneflow_funcs:
    if not funcs_name.startswith("_") and funcs_name not in internal_oneflow_funcs:
        if hasattr(torch, funcs_name):
            mapping_dict[eval("torch." + funcs_name)] = "oneflow." + funcs_name

oneflow_nn_functional_funcs = dir(oneflow.nn.functional)
for funcs_name in oneflow_nn_functional_funcs:
    if not funcs_name.startswith("_") and funcs_name not in internal_oneflow_funcs:
        if hasattr(torch.nn.functional, funcs_name):
            mapping_dict[eval("torch.nn.functional." + funcs_name)] = "oneflow.nn.functional." + funcs_name


def to_of_transform(
    gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            node.target = eval(mapping_dict[node.target])
        elif node.op == "call_method":
            if hasattr(torch.Tensor, node.target):
                pass
            else:
                raise NotImplementedError

    gm.graph.lint()
    gm.recompile()
    return gm
