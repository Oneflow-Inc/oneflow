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
import oneflow_api


def oneflow_export(*api_names, **kwargs):
    def Decorator(func_or_class):
        new_api_names = list(api_names)
        if hasattr(func_or_class, "_ONEFLOW_API_TAG"):
            if func_or_class._ONEFLOW_API_TAG == "experimental_api":
                new_api_names = ["experimental." + n for n in new_api_names]
        else:
            new_api_names = ["experimental." + n for n in new_api_names] + new_api_names
        func_or_class._ONEFLOW_API = new_api_names
        return func_or_class

    return Decorator


def stable_api(func_or_class):
    func_or_class._ONEFLOW_API_TAG = "stable_api"
    return func_or_class


def experimental_api(func_or_class):
    func_or_class._ONEFLOW_API_TAG = "experimental_api"
    return func_or_class


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


def export_oneflow_api_internal_symbols(oneflow_api, internal_name, api_name):
    names = internal_name.split(".")
    api = oneflow_api
    for n in names:
        api = getattr(api, n)
    globals()[api_name] = api
    oneflow_export(api_name)(api)


internal_names_2_api_names = {
    "PlacementSymbol": "placement",
    "Size": "Size",
    "device": "device",
    "autograd.no_grad": "no_grad",
}

for internal_name, api_name in internal_names_2_api_names.items():
    export_oneflow_api_internal_symbols(oneflow_api, internal_name, api_name)
