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

from oneflow.autograd.autograd import backward, grad
from oneflow.autograd.autograd_function import Function
from oneflow.autograd.autograd_mode import (
    set_grad_enabled,
    enable_grad,
    inference_mode,
    is_grad_enabled,
    no_grad,
)
from . import graph

__all__ = [
    "backward",
    "grad",
    "Function",
    "set_grad_enabled",
    "enable_grad",
    "inference_mode",
    "is_grad_enabled",
    "no_grad",
]
