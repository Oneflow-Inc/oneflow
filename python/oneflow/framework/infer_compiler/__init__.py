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

try:
    import torch
except ImportError:
    print("You should install torch also when use `oneflow.framework.infer_compiler`.")

from .transform.custom_transform import register
from .utils.patch_for_compiler import *
from .with_fx_graph import fx_node_tranform
from .with_fx_interpreter import OneFlowInterpreter
from .with_oneflow_compile import compile_from_torch
from .with_oneflow_backend import oneflow_backend
