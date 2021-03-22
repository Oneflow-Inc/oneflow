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

import os
import sys
import getpass
import imp
import inspect
import socket
from contextlib import closing
import uuid
import unittest
import atexit
from tempfile import NamedTemporaryFile
import google.protobuf.text_format as pbtxt
import oneflow
import oneflow.python.framework.env_util as env_util
from oneflow.core.job.env_pb2 import EnvProto
from oneflow.python.oneflow_export import oneflow_export
from typing import Any, Dict, Callable
import subprocess


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


@oneflow_export("unittest.env.has_world_size")
def has_world_size():
    if os.getenv("ONEFLOW_TEST_WORLD_SIZE"):
        assert os.getenv(
            "ONEFLOW_TEST_WORLD_SIZE"
        ).isdigit(), "env var ONEFLOW_TEST_WORLD_SIZE must be num"
        return True
    else:
        return False


@oneflow_export("unittest.env.world_size")
def world_size():
    return int(os.getenv("ONEFLOW_TEST_WORLD_SIZE"))


@oneflow_export("unittest.env.device_num")
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
                master_port = os.getenv("ONEFLOW_TEST_MASTER_PORT")
                assert master_port, "env var ONEFLOW_TEST_MASTER_PORT not set"
                oneflow.env.ctrl_port(int(master_port))
                if enable_init_by_host_list():
                    oneflow.env.machine(node_list())
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
                else:
                    ctrl_port = os.getenv("ONEFLOW_TEST_CTRL_PORT")
                    config_rank_ctrl_port = -1
                    if ctrl_port:
                        config_rank_ctrl_port = int(ctrl_port)

                    if has_world_size():
                        config_world_size = world_size()
                    else:
                        config_world_size = 0

                    config_node_size = -1
                    env_node_size = os.getenv("ONEFLOW_TEST_NODE_SIZE")
                    if env_node_size:
                        config_node_size = int(env_node_size)

                    bootstrap_conf_list = oneflow.env.init_bootstrap_confs(
                        node_list(),
                        int(master_port),
                        config_world_size,
                        config_rank_ctrl_port,
                        config_node_size,
                    )

                    data_port = os.getenv("ONEFLOW_TEST_DATA_PORT")
                    if data_port:
                        oneflow.env.data_port(int(data_port))

                    ssh_port = os.getenv("ONEFLOW_TEST_SSH_PORT")
                    print("initializing worker...")
                    oneflow.deprecated.init_worker(
                        scp_binary=True,
                        use_uuid=True,
                        ssh_port=int(ssh_port),
                        bootstrap_conf_list=bootstrap_conf_list,
                    )
                    atexit.register(
                        oneflow.deprecated.delete_worker_by_bootstrap, ssh_port=ssh_port
                    )
                    _unittest_worker_initilized = True
        elif device_num() > 1 and enable_multi_process():
            oneflow_worker_path = os.getenv("ONEFLOW_WORKER_BIN")
            assert oneflow_worker_path is not None, "please set env ONEFLOW_WORKER_BIN"
            assert os.path.isfile(
                oneflow_worker_path
            ), "binary oneflow_worker not found, please check your environment variable ONEFLOW_WORKER_BIN, path: {}".format(
                oneflow_worker_path
            )
            master_port = find_free_port()
            oneflow.env.ctrl_port(master_port)
            config_world_size = device_num()
            bootstrap_conf_list = oneflow.env.init_bootstrap_confs(
                ["127.0.0.1"], master_port, config_world_size
            )
            env_proto = env_util.default_env_proto
            assert (
                len(env_proto.machine) == 1
                and env_proto.HasField("ctrl_bootstrap_conf") == 1
            )
            run_dir = os.getenv("HOME") + "/oneflow_temp/" + str(uuid.uuid1())
            run_dir = os.path.abspath(os.path.expanduser(run_dir))
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            for rank in range(1, config_world_size):
                worker_env_proto = EnvProto()
                worker_env_proto.CopyFrom(env_proto)
                worker_env_proto.ctrl_bootstrap_conf.rank = rank
                worker_env_proto.cpp_logging_conf.log_dir = (
                    run_dir + "/log_" + str(rank)
                )
                env_file = NamedTemporaryFile(delete=False)
                if sys.version_info >= (3, 0):
                    env_file.write(pbtxt.MessageToString(worker_env_proto).encode())
                else:
                    env_file.write(pbtxt.MessageToString(worker_env_proto))
                env_file.close()
                if not os.path.exists(run_dir + "/log_" + str(rank)):
                    os.mkdir(run_dir + "/log_" + str(rank))
                os.system(
                    "cp "
                    + env_file.name
                    + " "
                    + run_dir
                    + "/log_"
                    + str(rank)
                    + "/env_proto_"
                    + str(rank)
                    + ".proto"
                )
                oneflow_cmd = (
                    oneflow_worker_path
                    + " -env_proto="
                    + run_dir
                    + "/log_"
                    + str(rank)
                    + "/"
                    + "env_proto_"
                    + str(rank)
                    + ".proto "
                )
                subprocess.Popen(
                    oneflow_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=True,
                )
                os.remove(env_file.name)
            atexit.register(
                oneflow.deprecated.delete_worker_of_multi_process, run_dir=run_dir
            )

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


@oneflow_export("unittest.skip_unless_1n3d")
def skip_unless_1n3d():
    return skip_unless(1, 3)


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
