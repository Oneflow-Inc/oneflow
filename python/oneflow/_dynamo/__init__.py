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

# Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/torch/_dynamo/__init__.py
__all__ = [
    "allow_in_graph",
]

def allow_in_graph(fn):
    """
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    warnings.warn(
        "The oneflow._dynamo.allow_in_graph interface is just to align the torch._dynamo.allow_in_graph interface and has no practical significance."
    )
    return fn

