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
from .torch_fx_to_oneflow import to_of_transform


def register_ofrt():
    from typing import List, Optional, Dict, Any
    import torch
    from torch import fx
    from torch._dynamo.backends.registry import register_backend
    from torch._dynamo.backends.common import fake_tensor_unsupported

    @register_backend
    @fake_tensor_unsupported
    def ofrt(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        # TODO(): fxGraphModule to nn.Graph
        print("my_compiler() called with FX graph:")
        gm.graph.print_tabular()
        print("gm ", gm)

        import oneflow as flow

        of_gm = to_of_transform(gm)

        enable_graph = bool(os.environ.get("ofrt_enable_graph", False))

        if not enable_graph:
            oneflow_fn = of_gm.forward
        else:

            @flow.nn.Graph.trace
            def oneflow_fn(inputs):
                outs = of_gm.forward(inputs)
                return outs

            oneflow_fn.debug(1)

        def from_to_torch(inputs):
            flow_inputs = flow.utils.tensor.from_torch(inputs)
            flow_outs = oneflow_fn(flow_inputs)
            # TODO(): general output process
            outs = flow.utils.tensor.to_torch(flow_outs[0])
            return (outs,)

        return from_to_torch
