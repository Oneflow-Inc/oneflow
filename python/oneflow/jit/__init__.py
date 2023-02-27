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
import warnings
from typing import Any, Dict, List, Set, Tuple, Union, Callable


def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    warnings.warn(
        "The oneflow.jit.script interface is just to align the torch.jit.script interface and has no practical significance."
    )
    return obj


def ignore(drop=False, **kwargs):
    warnings.warn(
        "The oneflow.jit.ignore interface is just to align the torch.jit.ignore interface and has no practical significance."
    )

    def decorator(fn):
        return fn

    return decorator


def unused(fn):
    warnings.warn(
        "The oneflow.jit.unused interface is just to align the torch.jit.unused interface and has no practical significance."
    )

    return fn


def _script_if_tracing(fn):
    warnings.warn(
        "The oneflow.jit._script_if_tracing interface is just to align the torch.jit._script_if_tracing interface and has no practical significance."
    )

    return fn


def _overload_method(fn):
    warnings.warn(
        "The oneflow.jit._overload_method interface is just to align the torch.jit._overload_method interface and has no practical significance."
    )

    return fn


def is_scripting():
    return False


def is_tracing():
    return False
