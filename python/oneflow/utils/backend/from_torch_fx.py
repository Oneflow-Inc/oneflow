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
import os
import torch
import oneflow as flow
from torch import fx
from typing import Dict, Any, Dict, Tuple


def fx_tranform(gm):

    of_gm = to_of_transform(gm)

    enable_graph = os.getenv("ofrt_enable_graph", "False").lower() in (
        "true",
        "1",
        "t",
    )

    if not enable_graph:
        oneflow_fn = of_gm.forward
    else:
        class OfGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = of_gm
            
            def build(self, *args, **kwargs):
                return self.m(*args, **kwargs)
        
        of_g = OfGraph()
        of_g.debug(0)
        oneflow_fn = lambda *args, **kwargs: of_g(*args, **kwargs)

    return oneflow_fn

def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def _replace_node_module(
    node: torch.fx.Node, modules: Dict[str, Any], new_module: flow.nn.Module
):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def _get_module(origin_mod):
    linear = flow.nn.Linear(3, 8, False)
    linear = linear.to("cuda")
    flow.nn.init.constant_(linear.weight, 2.3)
    return linear

def _to_of_transform(
    gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
) -> torch.fx.GraphModule:
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.relu:
                node.target = flow.relu
        elif node.op == "call_module":
            print(node.format_node())
            if type(modules[node.target] is torch.nn.Linear):
                linear = modules[node.target]
                print(linear)
                _replace_node_module(node, modules, _get_module(linear))

    gm.graph.lint()
    gm.recompile()
    for node in gm.graph.nodes:
        print(node.format_node())
    return gm


def to_of_transform(
    gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
) -> torch.fx.GraphModule:
    name2node = {}
    name2obj = {}
    of_g = flow.fx.Graph()
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        print(node.format_node())
        if node.op == "call_function":
            if node.target == torch.relu:
                node.target = flow.relu
        elif node.op == "call_module":
            if type(modules[node.target] is torch.nn.Linear):
                linear = modules[node.target]
                name2obj[node.target] = _get_module(linear)
                of_node = of_g.create_node('call_module', node.target, args=(name2node[node.args[0].name],))
                name2node[node.name] = of_node
        elif node.op == "call_method":
            ...
        elif node.op == "get_attr":
            ...
        elif node.op == "placeholder":
            of_node = of_g.create_node('placeholder', node.target)
            name2node[node.name] = of_node
        elif node.op == "output":
            of_g.output((name2node[node.args[0][0].name],))
        else:
            raise ValueError(f"not valid node type{node.foramt_node()}")
    print("\n new of graph", of_g.print_tabular())
    for of_node in of_g.nodes:
        print(of_node.format_node())
    
    of_gm = flow.fx.GraphModule(name2obj, of_g)
    of_gm.graph.lint()
    of_gm.recompile()
    return of_gm