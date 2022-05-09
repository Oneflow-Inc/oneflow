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
import atexit
import imp
import os
import socket
import subprocess
import sys
import unittest
import uuid
import doctest
from contextlib import closing
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict

import google.protobuf.text_format as pbtxt

import oneflow
import oneflow.env
import oneflow.sysconfig
from oneflow.core.job.env_pb2 import EnvProto


def register_test_cases(
    scope: Dict[str, Any],
    directory: str,
    filter_by_num_nodes: Callable[[bool], int],
    base_class: unittest.TestCase = unittest.TestCase,
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


def num_nodes_required(num_nodes: int) -> Callable[[Callable], Callable]:
    def Decorator(f):
        f.__oneflow_test_case_num_nodes_required__ = num_nodes
        return f

    return Decorator


def _GetNumOfNodes(func):
    if hasattr(func, "__oneflow_test_case_num_nodes_required__") == False:
        return 1
    return getattr(func, "__oneflow_test_case_num_nodes_required__")


def eager_execution_enabled():
    return os.getenv("ONEFLOW_TEST_ENABLE_EAGER") == "1"


def typing_check_enabled():
    return os.getenv("ONEFLOW_TEST_ENABLE_TYPING_CHECK") == "1"


def node_list():
    node_list_str = os.getenv("ONEFLOW_TEST_NODE_LIST")
    assert node_list_str
    return node_list_str.split(",")


def has_node_list():
    if os.getenv("ONEFLOW_TEST_NODE_LIST"):
        return True
    else:
        return False


def node_size():
    node_num_from_env = os.getenv("ONEFLOW_TEST_NODE_NUM", None)
    if node_num_from_env:
        return int(node_num_from_env)
    elif has_node_list():
        node_list_from_env = node_list()
        return len(node_list_from_env)
    else:
        return 1


def has_world_size():
    return True


def world_size():
    return oneflow.env.get_world_size()


def device_num():
    device_num_str = os.getenv("ONEFLOW_TEST_DEVICE_NUM")
    if device_num_str:
        return int(device_num_str)
    else:
        return 1


def enable_init_by_host_list():
    return os.getenv("ONEFLOW_TEST_ENABLE_INIT_BY_HOST_LIST") == "1"


def enable_multi_process():
    return os.getenv("ONEFLOW_TEST_MULTI_PROCESS") == "1"


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


_unittest_worker_initilized = False


def worker_agent_port():
    port_txt = os.getenv("ONEFLOW_TEST_WORKER_AGENT_PORT")
    if port_txt:
        return int(port_txt)
    else:
        return None


def worker_agent_authkey():
    key = os.getenv("ONEFLOW_TEST_WORKER_AGENT_AUTHKEY")
    assert key
    return key


def use_worker_agent():
    return worker_agent_port() is not None


def cast(conn=None, cmd=None, msg=None):
    cmd = "cast/" + cmd
    print("[unittest]", f"[{cmd}]", msg)
    conn.send(cmd.encode())
    conn.send(msg.encode())


def call(conn=None, cmd=None, msg=None):
    cmd = "call/" + cmd
    print("[unittest]", f"[{cmd}]", msg)
    conn.send(cmd.encode())
    msg_ = ""
    if msg is not None:
        msg_ = msg
    conn.send(msg_.encode())
    return conn.recv().decode()


TestCase = unittest.TestCase


def skip_unless(n, d):
    if (n > 1 or d > 1) and oneflow.sysconfig.has_rpc_backend_grpc() == False:
        return unittest.skip(
            "requires multi node rpc backend when node_size > 1 and device_num > 1"
        )
    if node_size() == n and device_num() == d:
        return lambda func: func
    else:
        return unittest.skip(
            "only runs when node_size is {} and device_num is {}".format(n, d)
        )


def skip_unless_1n1d():
    return skip_unless(1, 1)


def skip_unless_1n2d():
    return skip_unless(1, 2)


def skip_unless_1n4d():
    return skip_unless(1, 4)


def skip_unless_2n1d():
    return skip_unless(2, 1)


def skip_unless_2n2d():
    return skip_unless(2, 2)


def skip_unless_2n4d():
    return skip_unless(2, 4)


class CondSkipChecker(doctest.OutputChecker):
    def __init__(self, check_flags):
        self._check_flags = check_flags

    def check_output(self, want, got, optionflags):
        # default check_output without flag
        if optionflags == 0:
            return super(CondSkipChecker, self).check_output(want, got, optionflags)

        target_rank_list = [bool(flag & optionflags) for flag in self._check_flags]
        # wrong flag will be handled before here, so any(target_rank_list) is True
        # not target rank
        if target_rank_list.index(True) != oneflow.env.get_rank():
            return True
        elif target_rank_list.index(True) == oneflow.env.get_rank():
            return super(CondSkipChecker, self).check_output(want, got, optionflags)


def check_multi_rank_docstr(module):
    # supply customized flag ONLY_CHECK_RANK_{x} for docstr
    check_flags = [
        doctest.register_optionflag(f"ONLY_CHECK_RANK_{i}")
        for i in range(oneflow.env.get_world_size())
    ]
    finder = doctest.DocTestFinder()
    runner = doctest.DebugRunner(CondSkipChecker(check_flags))
    for test in finder.find(module, module.__name__):
        runner.run(test)
