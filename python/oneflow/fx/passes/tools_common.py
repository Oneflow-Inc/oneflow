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
from typing import List, Tuple, Union, Dict, Any, Set
from dataclasses import dataclass

import oneflow as flow
import oneflow.fx
from oneflow.fx.node import _get_qualified_name


Tensors = Union[Tuple[flow.Tensor], List[flow.Tensor]]
TensorOrTensors = Union[flow.Tensor, Tensors]
NodeList = List[flow.fx.Node]
NodeSet = Set[flow.fx.Node]
Names = List[str]
CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}


def typename(o):
    if isinstance(o, flow.Tensor):
        return o.type()

    module = ""
    class_name = ""
    if (
        hasattr(o, "__module__")
        and o.__module__ != "builtins"
        and o.__module__ != "__builtin__"
        and o.__module__ is not None
    ):
        module = o.__module__ + "."

    if hasattr(o, "__qualname__"):
        class_name = o.__qualname__
    elif hasattr(o, "__name__"):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name


def get_node_target(submodules: Dict[str, flow.nn.Module], node: flow.fx.Node) -> str:
    """
    Given a `node` returns its target typename.

    For "call_method" node, return node.target which is the name of that method being called.
    This could potential lead to conflict but should be okay because normally it's on a tensor.

    For "call_function" node, return typename of node.target.

    For "call_module" node, return typename of the module that node.target point to.

    If seeing "_VariableFunctionsClass" in the target name string, it will be replaced by
    "flow". e.g. _VariableFunctionsClass.relu would become flow.relu.
    """

    assert node.op in CALLABLE_NODE_OPS, (
        "Expect op types of " + ", ".join(CALLABLE_NODE_OPS) + f", but found {node.op}"
    )

    if node.op == "call_module":
        assert isinstance(node.target, str)
        return typename(submodules[node.target])
    elif node.op == "call_function":
        target: Any = node.target
        return (
            f"acc_ops.{target.__name__}"
            if target.__module__ is not None and "acc_ops" in target.__module__
            else _get_qualified_name(target)
        )
    else:
        assert isinstance(node.target, str)
        return node.target


def is_node_output_tensor(node: flow.fx.Node) -> bool:
    """Checks if the node output produces a Tensor or not.

    NOTE: This requires to run `ShapeProp` on the containing fx graph before
    calling this function. This is because it works by checking the `type`
    metadata on the node. This metadata is produced by the `ShapeProp`.
    """
    type_ = node.meta.get("type", None)
    return type_ is not None and issubclass(type_, flow.Tensor)
