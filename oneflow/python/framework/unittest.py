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


@oneflow_export("unittest.register_test_cases")
def register_test_cases(
    scope: Dict[str, Any],
    directory: str,
    filter_by_num_nodes: Callable[[bool], int],
    base_class: unittest.TestCase = unittest.TestCase,
    test_case_mixin=_ClearDefaultSession,
) -> None:
    def FilterTestPyFile(f):
        return (
            os.path.isfile(os.path.join(directory, f))
            and f.endswith(".py")
            and f.startswith("test")
        )

    def FilterMethodName(module, name):
        method = getattr(module, name)
        return (
            name.startswith("test")
            and callable(method)
            and filter_by_num_nodes(_GetNumOfNodes(method))
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


def _GetNumOfNodes(func):
    if hasattr(func, "__oneflow_test_case_num_nodes_required__") == False:
        return 1
    return getattr(func, "__oneflow_test_case_num_nodes_required__")


@oneflow_export("unittest.env.eager_execution_enabled")
def eager_execution_enabled():
    return os.getenv("ONEFLOW_TEST_ENABLE_EAGER") == "1"


@oneflow_export("unittest.env.typing_check_enabled")
def typing_check_enabled():
    return os.getenv("ONEFLOW_TEST_ENABLE_TYPING_CHECK") == "1"


@oneflow_export("unittest.env.node_list")
def node_list():
    node_list_str = os.getenv("ONEFLOW_TEST_NODE_LIST")
    assert node_list_str
    return node_list_str.split(",")


@oneflow_export("unittest.env.has_node_list")
def has_node_list():
    if os.getenv("ONEFLOW_TEST_NODE_LIST"):
        return True
    else:
        return False


@oneflow_export("unittest.env.node_size")
def node_size():
    if has_node_list():
        node_list_from_env = node_list()
        return len(node_list_from_env)
    else:
        return 1


@oneflow_export("unittest.env.gpu_device_num")
def gpu_device_num():
    gpu_num_str = os.getenv("ONEFLOW_TEST_GPU_DEVICE_NUM")
    if gpu_num_str:
        return int(gpu_num_str)
    else:
        return 1


_unittest_env_initilized = False


@oneflow_export("unittest.TestCase")
class TestCase(unittest.TestCase):
    def setUp(self):

        if has_node_list():
            assert node_size() > 1
            oneflow.env.machine(node_list())

            data_port = os.getenv("ONEFLOW_TEST_DATA_PORT")
            assert data_port, "env var ONEFLOW_TEST_DATA_PORT not set"
            oneflow.env.data_port(int(dport))

            ctrl_port = os.getenv("ONEFLOW_TEST_CTRL_PORT")
            assert ctrl_port, "env var ONEFLOW_TEST_CTRL_PORT not set"
            oneflow.env.ctrl_port(int(ctrl_port))

            oneflow.deprecated.init_worker(scp_binary=True, use_uuid=True)

        global _unittest_env_initilized
        if _unittest_env_initilized == False:
            oneflow.env.init()
            _unittest_env_initilized = True
            if has_node_list():
                atexit.register(flow.deprecated.delete_worker)

        oneflow.clear_default_session()
        oneflow.enable_eager_execution(eager_execution_enabled())
        oneflow.experimental.enable_typing_check(typing_check_enabled())


@oneflow_export("unittest.OneGpuTestCase")
class TestCase_1n1c(TestCase):
    def setUp(self):
        if node_size() == 1 and gpu_device_num() == 1:
            super().setUp()
        else:
            skip_reason = "only runs when node_size is 1 and gpu_device_num is 1: {!r}"
            self.skipTest(skip_reason)
