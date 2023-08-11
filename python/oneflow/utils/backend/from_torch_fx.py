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
import oneflow
import torch
from torch import fx


def fx_tranform(gm):
    import oneflow as flow

    of_gm = to_of_transform(gm)

    enable_graph = os.getenv("ofrt_enable_graph", "False").lower() in (
        "true",
        "1",
        "t",
    )

    if not enable_graph:
        oneflow_fn = of_gm.forward
    else:
        @flow.nn.Graph.trace
        def oneflow_fn(inputs):
            outs = of_gm.forward(inputs)
            return outs

        oneflow_fn.debug(1)
    return oneflow_fn

def to_of_transform(
    gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.relu:
                node.target = oneflow.relu

    gm.graph.lint()
    gm.recompile()
    return gm
