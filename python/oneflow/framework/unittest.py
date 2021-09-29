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
import oneflow.framework.env_util as env_util
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
    if oneflow.env.is_multi_client():
        return True
    if os.getenv("ONEFLOW_TEST_WORLD_SIZE"):
        assert os.getenv(
            "ONEFLOW_TEST_WORLD_SIZE"
        ).isdigit(), "env var ONEFLOW_TEST_WORLD_SIZE must be num"
        return True
    else:
        return False


def world_size():
    if oneflow.env.is_multi_client():
        return oneflow.env.get_world_size()
    return int(os.getenv("ONEFLOW_TEST_WORLD_SIZE"))


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


def launch_worker_via_agent(host=None, env_proto=None):
    print("[unittest]", "launching worker via agent at", host)
    from multiprocessing.connection import Client

    address = ("localhost", worker_agent_port())
    conn = Client(address, authkey=worker_agent_authkey().encode())
    cast(conn=conn, cmd="host", msg=host)
    cast(conn=conn, cmd="env_proto", msg=pbtxt.MessageToString(env_proto))
    assert call(conn=conn, cmd="start_worker") == "ok"
    print("[unittest]", "worker launched via agent at", host)
    conn.close()


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
                data_port = os.getenv("ONEFLOW_TEST_DATA_PORT")
                if data_port:
                    oneflow.env.data_port(int(data_port))
                if enable_init_by_host_list():
                    oneflow.env.machine(node_list())
                    data_port = os.getenv("ONEFLOW_TEST_DATA_PORT")
                    print("initializing worker...")
                    for machine in env_util.default_env_proto.machine:
                        if machine.id == 0:
                            pass
                        else:
                            launch_worker_via_agent(
                                host=machine.addr, env_proto=env_util.default_env_proto
                            )
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
                    worker_env_proto = EnvProto()
                    worker_env_proto.CopyFrom(env_util.default_env_proto)
                    worker_env_proto.ClearField("ctrl_bootstrap_conf")
                    for bootstrap_conf in bootstrap_conf_list:
                        if bootstrap_conf.rank == 0:
                            continue
                        assert bootstrap_conf.HasField("host")
                        worker_env_proto.ctrl_bootstrap_conf.CopyFrom(bootstrap_conf)
                        launch_worker_via_agent(
                            host=bootstrap_conf.host, env_proto=worker_env_proto
                        )
                _unittest_worker_initilized = True
        elif device_num() > 1 and enable_multi_process():
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
                    "python3 -m oneflow --start_worker"
                    + " --env_proto="
                    + run_dir
                    + "/log_"
                    + str(rank)
                    + "/"
                    + "env_proto_"
                    + str(rank)
                    + ".proto"
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
            env_util.api_env_init()
            _unittest_env_initilized = True


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
