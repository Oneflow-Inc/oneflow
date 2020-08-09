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
import inspect
import re
import collections

import oneflow.python.lib.core.enable_if as enable_if_util
import oneflow.python.lib.core.traceinfo as traceinfo
from oneflow.python.lib.core.high_order_bool import always_true


def oneflow_export(*api_names, **kwargs):
    def Decorator(func_or_class):
        func_or_class._ONEFLOW_API = api_names
        return func_or_class

    return Decorator


_DEPRECATED = set()


def oneflow_deprecate(*api_names, **kwargs):
    def Decorator(func_or_class):
        _DEPRECATED.add(func_or_class)
        return func_or_class

    return Decorator


@oneflow_export("is_deprecated")
def is_deprecated(func_or_class):
    return (
        isinstance(func_or_class, collections.Hashable) and func_or_class in _DEPRECATED
    )
