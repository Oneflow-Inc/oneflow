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

def _get_output_names(node):
    return [output.debugName() for output in node.outputs()]

def _get_input_names(node_or_graph):
    return [inp.debugName() for inp in node_or_graph.inputs()]

def script_tranform(gm, example_inputs):
    import oneflow as flow
    import pdb; pdb.set_trace()
    print("transform from torch script")

    jit_mod = torch.jit.trace(gm, tuple(example_inputs))
    print("jit mod graph ", jit_mod.graph)
    torch_graph = jit_mod.graph.copy()

    nodes = torch_graph.nodes()
    for node in nodes:
        print("===")
        print("node: ", node)
        operator = node.kind()
        input_names = _get_input_names(node)
        output_names = _get_output_names(node)
        print("in: ", input_names)
        print("out: ", output_names)
        if operator == "prim::relu":
            print("prim::relu")
        elif operator == "prim::TupleConstruct":
            print("prim::TupleConstruct")
        else:
            print(operator)



    enable_graph = os.getenv("ofrt_enable_graph", "False").lower() in (
        "true",
        "1",
        "t",
    )
    return gm

    if not enable_graph:
        oneflow_fn = of_gm.forward
    else:
        @flow.nn.Graph.trace
        def oneflow_fn(inputs):
            outs = of_gm.forward(inputs)
            return outs

        oneflow_fn.debug(1)
    return oneflow_fn
