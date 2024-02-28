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
from oneflow.framework.args_tree import ArgsTree
from .with_fx_graph import fx_node_tranform
from .with_fx_interpreter import OneFlowInterpreter

def oneflow_backend(gm, example_inputs, *args, **kwargs):
    with_interp = os.getenv(
        "ONEDIFF_INFER_COMPILER_USE_INTERPRETER", "False"
    ).lower() in ("true", "1", "t",)
    if not with_interp:
        transformed_fn = fx_node_tranform(gm)

    def wrapped_forward(*args, **kwargs):
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                return flow.utils.tensor.from_torch(value.contiguous())
            else:
                return value

        args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
        out = args_tree.map_leaf(input_fn)
        args = out[0]
        if with_interp:
            output = OneFlowInterpreter(gm, garbage_collect_values=False).run(
                *args, **kwargs
            )
        else:
            output = transformed_fn(*args, **kwargs)
        if isinstance(output, tuple):
            return tuple(flow.utils.tensor.to_torch(i) for i in output)
        return flow.utils.tensor.to_torch(output)
    
    return wrapped_forward