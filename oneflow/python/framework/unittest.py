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
import atexit
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


@oneflow_export("unittest.env.device_num")
def device_num():
    device_num_str = os.getenv("ONEFLOW_TEST_DEVICE_NUM")
    if device_num_str:
        return int(device_num_str)
    else:
        return 1


_unittest_env_initilized = False
_unittest_worker_initilized = False


@oneflow_export("unittest.TestCase")
class TestCase(unittest.TestCase):
    def setUp(self):
        global _unittest_env_initilized
        global _unittest_worker_initilized

        if has_node_list():
            assert node_size() > 1

            if _unittest_worker_initilized == False:
                oneflow.env.machine(node_list())

                ctrl_port = os.getenv("ONEFLOW_TEST_CTRL_PORT")
                assert ctrl_port, "env var ONEFLOW_TEST_CTRL_PORT not set"
                oneflow.env.ctrl_port(int(ctrl_port))

                data_port = os.getenv("ONEFLOW_TEST_DATA_PORT")
                if data_port:
                    oneflow.env.data_port(int(data_port))

                ssh_port = os.getenv("ONEFLOW_TEST_SSH_PORT")
                print("initializing worker...")
                oneflow.deprecated.init_worker(
                    scp_binary=True, use_uuid=True, ssh_port=int(ssh_port)
                )
                atexit.register(oneflow.deprecated.delete_worker, ssh_port=ssh_port)
                _unittest_worker_initilized = True

        log_dir = os.getenv("ONEFLOW_TEST_LOG_DIR")
        if log_dir:
            oneflow.env.log_dir(log_dir)

        if _unittest_env_initilized == False:
            oneflow.env.init()
            _unittest_env_initilized = True

        oneflow.clear_default_session()
        oneflow.enable_eager_execution(eager_execution_enabled())
        oneflow.experimental.enable_typing_check(typing_check_enabled())


def skip_unless(n, d):
    if node_size() == n and device_num() == d:
        return lambda func: func
    else:
        return unittest.skip(
            "only runs when node_size is {} and device_num is {}".format(n, d)
        )


@oneflow_export("unittest.skip_unless_1n1d")
def skip_unless_1n1d():
    return skip_unless(1, 1)


@oneflow_export("unittest.skip_unless_1n2d")
def skip_unless_1n2d():
    return skip_unless(1, 2)


@oneflow_export("unittest.skip_unless_1n4d")
def skip_unless_1n4d():
    return skip_unless(1, 4)


@oneflow_export("unittest.skip_unless_2n1d")
def skip_unless_2n1d():
    return skip_unless(2, 1)


@oneflow_export("unittest.skip_unless_2n2d")
def skip_unless_2n2d():
    return skip_unless(2, 2)


@oneflow_export("unittest.skip_unless_2n4d")
def skip_unless_2n4d():
    return skip_unless(2, 4)
