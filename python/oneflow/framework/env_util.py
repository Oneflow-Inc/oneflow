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
import os
import socket
import traceback
from contextlib import closing
import warnings

import oneflow._oneflow_internal
import oneflow.core.control.ctrl_bootstrap_pb2 as ctrl_bootstrap_pb
import oneflow.core.job.env_pb2 as env_pb
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.framework.c_api_util as c_api_util


def api_all_device_placement(device_type: str) -> oneflow._oneflow_internal.placement:
    r"""
    oneflow.env.all_device_placement(device_type) -> oneflow.placement

    Returns a placement that contains all available devices.

    Note:
        It is recommended to use `oneflow.placement.all` instead of this function.

    Args:
        device_type (str): cuda or cpu

    For examples:

    .. code-block:: python

        # Runs on 4 ranks
        import oneflow as flow

        p = flow.env.all_device_placement("cuda") # oneflow.placement(type="cuda", ranks=[0, 1, 2, 3])
        p = flow.env.all_device_placement("cpu") # oneflow.placement(type="cpu", ranks=[0, 1, 2, 3])

    """
    return oneflow.placement.all(device_type)


def check_non_localhost_proxy_and_print_warning():
    for env_var_name in ["http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"]:
        env_var_value = os.getenv(env_var_name)
        if (
            env_var_value is not None
            and (not "://localhost" in env_var_value)
            and (not "://127.0.0.1" in env_var_value)
            and (not env_var_value.startswith("localhost"))
            and (not env_var_value.startswith("127.0.0.1"))
        ):
            print(
                f"Proxy through another machine ({env_var_value}) is incompatible with OneFlow. Please unset them by `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`"
            )
            break


def create_env():
    """create environment

    Returns:
        Env: [description]
    """
    global default_env_proto
    assert len(default_env_proto.machine) > 0
    CompleteEnvProto(default_env_proto)
    if default_env_proto.ctrl_bootstrap_conf.world_size > 1:
        check_non_localhost_proxy_and_print_warning()
    return c_api_util.GetEnvContext(default_env_proto)


def CompleteEnvProto(env_proto):
    _UpdateDefaultEnvProtoByMultiClientEnvVars(env_proto)
    if env_proto.HasField("ctrl_port") == False:
        if len(env_proto.machine) == 1:
            env_proto.ctrl_port = _FindFreePort()
        else:
            raise ValueError(
                "a ctrl_port is required if running multi-node, set it with 'oneflow.env.ctrl_port([YOUR PORT])'"
            )


def _MakeMachine(machines):
    if isinstance(machines, str):
        machines = [machines]
    rp_machine = env_pb.EnvProto().machine
    for m_data in machines:
        m = rp_machine.add()
        if isinstance(m_data, str):
            m.addr = m_data
        elif isinstance(m_data, dict):
            if "addr" in m_data:
                m.addr = m_data["addr"]
            if "ctrl_port_agent" in m_data:
                m.ctrl_port_agent = m_data["ctrl_port_agent"]
            if "data_port_agent" in m_data:
                m.data_port_agent = m_data["data_port_agent"]
        else:
            raise NotImplementedError
    id = 0
    addrs_for_check = set()
    for m in rp_machine:
        m.id = id
        id += 1
        assert m.addr not in addrs_for_check
        addrs_for_check.add(m.addr)
    return rp_machine


def _MakeBootstrapConf(bootstrap_info: dict):
    global config_master_addr
    assert config_master_addr.HasField("host"), "must config master host first"
    assert config_master_addr.HasField("port"), "must config master port first"
    assert config_world_size != 0, "must config world size first"
    bootstrap_conf = ctrl_bootstrap_pb.BootstrapConf()
    bootstrap_conf.master_addr.CopyFrom(config_master_addr)
    bootstrap_conf.world_size = config_world_size
    assert "rank" in bootstrap_info
    bootstrap_conf.rank = bootstrap_info["rank"]
    if "host" in bootstrap_info:
        bootstrap_conf.host = bootstrap_info["host"]
    global config_bootstrap_ctrl_port
    if config_bootstrap_ctrl_port != 0:
        bootstrap_conf.ctrl_port = config_bootstrap_ctrl_port
    global config_node_size
    if config_node_size != 0:
        bootstrap_conf.node_size = config_node_size
    return bootstrap_conf


def _DefaultEnvProto():
    env_proto = env_pb.EnvProto()
    machine = env_proto.machine.add()
    machine.id = 0
    machine.addr = "127.0.0.1"
    return env_proto


def _FindFreePort():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def CheckAndWarnAbnormalEnvVars():
    env_var_names = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    env_var_without_value = [x for x in env_var_names if os.getenv(x) is None]
    env_var_with_value = [x for x in env_var_names if os.getenv(x) is not None]
    if len(env_var_with_value) != 0 and len(env_var_without_value) != 0:
        warnings.warn(
            f"Among four environment variables required for distributed training, only {', '.join('`{0}`'.format(x) for x in env_var_with_value)} are set, but {', '.join('`{0}`'.format(x) for x in env_var_without_value)} are not set."
        )


def _UpdateDefaultEnvProtoByMultiClientEnvVars(env_proto):
    def str2int(env_config):
        return int(env_config)

    bootstrap_conf = ctrl_bootstrap_pb.BootstrapConf()
    master_addr = ctrl_bootstrap_pb.Address()
    master_addr.host = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_addr.port = str2int(os.getenv("MASTER_PORT", _FindFreePort()))
    bootstrap_conf.master_addr.CopyFrom(master_addr)
    bootstrap_conf.world_size = str2int(os.getenv("WORLD_SIZE", 1))
    bootstrap_conf.rank = str2int(os.getenv("RANK", 0))
    env_proto.ctrl_bootstrap_conf.CopyFrom(bootstrap_conf)
    cpp_logging_conf = env_pb.CppLoggingConf()
    if os.getenv("GLOG_log_dir"):
        cpp_logging_conf.log_dir = os.getenv("GLOG_log_dir")
    if os.getenv("GLOG_logtostderr"):
        cpp_logging_conf.logtostderr = str2int(os.getenv("GLOG_logtostderr"))
    if os.getenv("GLOG_logbuflevel"):
        cpp_logging_conf.logbuflevel = str2int(os.getenv("GLOG_logbuflevel"))
    if os.getenv("GLOG_minloglevel"):
        cpp_logging_conf.minloglevel = str2int(os.getenv("GLOG_minloglevel"))
    env_proto.cpp_logging_conf.CopyFrom(cpp_logging_conf)


class EnvHolder(object):
    def __init__(self):
        CheckAndWarnAbnormalEnvVars()
        self._env_cxt = create_env()
        self._shutting_down = [False]

    def is_shutting_down(self):
        """
        Whether the interpreter is currently shutting down.
        For use in finalizers, __del__ methods, and similar; it is advised
        to early bind this function rather than look it up when calling it,
        since at shutdown module globals may be cleared.

        Please refer to: https://github.com/Oneflow-Inc/OneTeam/issues/1219#issuecomment-1092370402
        This solution is obtained from cupy code: https://github.com/cupy/cupy/pull/2809
        """
        return self._shutting_down[0]

    def switch_to_shutting_down(self, is_normal_exit=True):
        self._shutting_down[0] = True
        self._env_cxt.SwitchToShuttingDownPhase(is_normal_exit)


def GetEnv():
    return EnvHolder()


device_tag2default_parallel_conf = {}
default_env_proto = _DefaultEnvProto()
config_master_addr = ctrl_bootstrap_pb.Address()
config_world_size = 0
config_bootstrap_ctrl_port = 0
config_node_size = 0
global_ctrl_bootstrap_confs = []
