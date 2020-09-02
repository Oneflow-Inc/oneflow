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
from __future__ import absolute_import

import imp
import inspect
import os
import unittest

import oneflow
from oneflow.python.oneflow_export import oneflow_export
from typing import Any, Dict, Callable


class _ClearDefaultSession(object):
    def setUp(self):
        oneflow.clear_default_session()
        oneflow.enable_eager_execution(False)

    def runTest(self):
        for f_name in dir(self):
            if f_name.startswith("test_"):
                getattr(self, f_name)()


@oneflow_export("unittest.register_test_cases")
def register_test_cases(
    scope: Dict[str, Any],
    directory: str,
    filter_by_num_nodes: Callable[[bool], int],
    base_class: unittest.TestCase = unittest.TestCase,
    test_case_mixin=_ClearDefaultSession,
    filter_by_num_gpus = 1 : Callable[[bool], int],
) -> None:
    def FilterTestPyFile(f):
        return (
            os.path.isfile(os.path.join(directory, f))
            and f.endswith(".py")
            and f.startswith("test")
            and f.startswith("test_util") == False
            and f.startswith("test_global_storage") == False
        )

    def FilterMethodName(module, name):
        method = getattr(module, name)
        return (
            name.startswith("test")
            and callable(method)
            and filter_by_num_nodes(_GetNumOfNodes(method))
            and filter_by_num_gpus(_GetNumOfGPUs(method))
        )

    onlytest_files = [f for f in os.listdir(directory) if FilterTestPyFile(f)]
    for f in onlytest_files:
        class_name = f[0:-3]
        module = imp.load_source(class_name, os.path.join(directory, f))
        test_func_names = [
            name for name in dir(module) if FilterMethodName(module, name)
        ]
        method_dict = {k: getattr(module, k) for k in test_func_names}
        scope[class_name] = type(class_name, (test_case_mixin, base_class), method_dict)


@oneflow_export("unittest.num_nodes_required")
def num_nodes_required(num_nodes: int) -> Callable[[Callable], Callable]:
    def Decorator(f):
        f.__oneflow_test_case_num_nodes_required__ = num_nodes
        return f

    return Decorator


@oneflow_export("unittest.num_gpus_per_node_required")
def num_gpus_per_node_required(num_gpus: int) -> Callable[[Callable], Callable]:
    def Decorator(f):
        f.__oneflow_test_case_num_gpus_per_node_required__ = num_gpus
        return f

    return Decorator


def _GetNumOfNodes(func):
    if hasattr(func, "__oneflow_test_case_num_nodes_required__") == False:
        return 1
    return getattr(func, "__oneflow_test_case_num_nodes_required__")


def _GetNumOfGPUs(func):
    if hasattr(func, "__oneflow_test_case_num_gpus_per_node_required__") == False:
        return 1
    return getattr(func, "__oneflow_test_case_num_gpus_per_node_required__")
