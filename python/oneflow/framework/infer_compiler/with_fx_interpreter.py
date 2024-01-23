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
from typing import Any, Dict, Tuple

import torch
from oneflow.framework.infer_compiler.transform.builtin_transform import torch2oflow

from .transform import ProxySubmodule, map_args


class OneFlowInterpreter(torch.fx.Interpreter):
    from torch.fx.node import Argument, Target

    def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        args, kwargs = map_args(args, kwargs)
        target = torch2oflow(target)
        return super().call_function(target, args, kwargs)

    def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        args, kwargs = map_args(args, kwargs)
        return super().call_method(target, args, kwargs)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        submod = self.fetch_attr(target)
        submod = ProxySubmodule(submod)
        return submod(*args, **kwargs)
