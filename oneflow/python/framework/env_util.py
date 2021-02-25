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

import socket
from contextlib import closing

import oneflow.core.control.ctrl_bootstrap_pb2 as ctrl_bootstrap_pb
import oneflow.core.job.env_pb2 as env_pb
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate
import oneflow_api
import traceback


@oneflow_export("enable_eager_execution")
def api_enable_eager_execution(val: bool = True) -> None:
    r"""If True, job will execute in eager mode, else use lazy mode(static graph).

    Args:
        val (bool, optional): Whether  eager execution or not.  Defaults to True.
    """
    return enable_if.unique([enable_eager_environment])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def enable_eager_environment(val=True):
    return oneflow_api.EnableEagerEnvironment(val)


@oneflow_export("env.init")
def api_env_init() -> bool:
    r"""Init environment for job

    Returns:
        bool: [description]
    """
    return enable_if.unique([env_init, do_nothing])()


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def env_init():
    global default_env_proto
    assert len(default_env_proto.machine) > 0
    CompleteEnvProto(default_env_proto)
    c_api_util.InitEnv(default_env_proto)
    if oneflow_api.CurrentMachineId() == 0:
        scope_util.InitScopeStack()
    else:
        exit(0)
    return True


@oneflow_export("env.current_resource", "current_resource")
def api_get_current_resource() -> resource_util.Resource:
    r"""Get current resources, such as:machine nums, cpu/gpu device nums,
            epoch network threed num, rdma params...

    Returns:
        resource_util.Resource: [description]
    """
    return enable_if.unique([get_current_resource])()


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def get_current_resource():
    return c_api_util.CurrentResource()


@oneflow_export("current_machine_id")
def api_get_current_machine_id():
    r"""Get machine id of current machine/node

    Returns:
        [type]: [description]
    """
    return enable_if.unique([get_current_machine_id])()


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def get_current_machine_id() -> int:
    return oneflow_api.CurrentMachineId()


@oneflow_export("env.machine")
def api_machine(*val: list) -> None:
    r"""Set machines' hostnames.

    For instance:

        oneflow.env.machine([{"addr": "192.168.1.1"}, {"addr": "192.168.1.2"}])

    Args:
        val:  `list`, `tuple` or multiple arguments of `dict`. First in the list is the master machine.
    """
    return enable_if.unique([machine, do_nothing])(*val)


# only used by CI
@oneflow_export("env.init_bootstrap_confs")
def api_init_bootstrap_confs(*val: list) -> None:
    return enable_if.unique([MakeBootstrapConfs, do_nothing])(*val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def machine(*val):

    del default_env_proto.machine[:]
    if len(val) == 1 and isinstance(val[0], (list, tuple)):
        val = val[0]
    default_env_proto.ClearField("machine")
    default_env_proto.machine.extend(_MakeMachine(val))


@oneflow_export("env.ctrl_port")
def api_ctrl_port(val: int) -> None:
    r"""Set port number used to control the execution across multiple machines. Same on every machine.

    Args:
        val: a port number accessible to peer machines
    """
    return enable_if.unique([ctrl_port, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def ctrl_port(val):
    assert type(val) is int
    default_env_proto.ctrl_port = val


@oneflow_export("env.master_host")
def api_master_port(host: str) -> None:
    r"""Set host of master node. Same on every machine.

    Args:
        val: a addr accessible to master node
    """
    return enable_if.unique([master_host, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def master_host(host):
    assert type(host) is str
    global config_master_addr
    config_master_addr.host = host


@oneflow_export("env.master_port")
def api_master_port(val: int) -> None:
    r"""Set port number of master node. Same on every machine.

    Args:
        val: a port number accessible to master node
    """
    return enable_if.unique([master_port, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def master_port(val):
    assert type(val) is int
    global config_master_addr
    config_master_addr.port = val


@oneflow_export("env.world_size")
def api_world_size(val: int) -> None:
    r"""Set number of rank in cluster. Same on every machine.

    Args:
        val: a port number of rank in cluster
    """
    return enable_if.unique([world_size, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def world_size(val):
    assert type(val) is int
    global config_world_size
    config_world_size = val


@oneflow_export("env.data_port")
def api_data_port(val: int) -> None:
    r"""Set port number used to data transfer among multiple machines. Same on every machine.

    Args:
        val: a port number accessible to peer machines
    """
    return enable_if.unique([data_port, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def data_port(val):
    assert type(val) is int
    default_env_proto.data_port = val


@oneflow_export("env.grpc_use_no_signal")
@oneflow_deprecate()
def api_grpc_use_no_signal(val: bool = True) -> None:
    r"""Set rpc use signal or not (deprecate)

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    print(
        "WARNING:",
        "oneflow.env.grpc_use_no_signal is deprecated, users no longer need to set rpc use signal or not. \n",
        traceback.format_stack()[-2],
    )
    return None


@oneflow_export("env.log_dir")
def api_log_dir(val: str) -> None:
    r"""Specify a dir to store OneFlow's logging files. If not specified, it is `./log` by default.

    Args:
        val (str): string , log file path
    """
    return enable_if.unique([log_dir, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def log_dir(val):
    assert type(val) is str
    default_env_proto.cpp_logging_conf.log_dir = val


@oneflow_export("env.logtostderr")
def api_logtostderr(val: int) -> None:
    r"""Set whether log messages go to stderr instead of logfiles

    Args:
        val (int): [description]
    """
    return enable_if.unique([logtostderr, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def logtostderr(val):
    assert type(val) is int
    default_env_proto.cpp_logging_conf.logtostderr = val


@oneflow_export("env.logbuflevel")
def api_logbuflevel(val: int) -> None:
    r"""Log messages at a level <= this flag are buffered.
            Log messages at a higher level are flushed immediately.

    Args:
        val (int): int, number of level
    """
    return enable_if.unique([logbuflevel, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def logbuflevel(val):
    assert type(val) is int
    default_env_proto.cpp_logging_conf.logbuflevel = val


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def do_nothing(*args, **kwargs):
    print("Nothing happened because environment has been initialized")
    return False


def CompleteEnvProto(env_proto):
    if len(env_proto.machine) == 1 and env_proto.HasField("ctrl_port") == False:
        env_proto.ctrl_port = _FindFreePort()


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
    assert config_world_size != 0, "must config world first"
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
    return bootstrap_conf


# only used by CI
@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def MakeBootstrapConfs(rank_host_list, master_port, world_size=0, ctrl_port=-1):
    r"""Set ctrl_bootstrap_conf' info.

    For instance:

        ONEFLOW_TEST_RANK_HOST_LIST=192.168.1.16,192.168.1.15 ONEFLOW_TEST_MASTER_PORT=43256
        ONEFLOW_TEST_WORLD_SIZE=2 ONEFLOW_TEST_RANK_CTRL_PORT=34527

    Args:
        val:  `list`, First in the list is the master machine.
    """
    if isinstance(rank_host_list, str):
        rank_host_list = [rank_host_list]
    global global_ctrl_bootstrap_confs
    assert len(global_ctrl_bootstrap_confs) == 0, "ctrl_bootstrap_conf has been inited"
    global config_master_addr
    config_master_addr.host = rank_host_list[0]
    config_master_addr.port = master_port
    global config_world_size
    if world_size == 0:
        config_world_size = len(rank_host_list)
    else:
        assert (world_size % len(rank_host_list)) == 0
        config_world_size = world_size
    global config_bootstrap_ctrl_port
    if ctrl_port != -1:
        config_bootstrap_ctrl_port = ctrl_port
    rank = 0
    for rank_host in rank_host_list:
        assert isinstance(rank_host, str)
        bootstrap_conf = _MakeBootstrapConf({"rank": rank, "host": rank_host})
        if rank == 0:
            global default_env_proto
            default_env_proto.ctrl_bootstrap_conf.CopyFrom(bootstrap_conf)
        global_ctrl_bootstrap_confs.append(bootstrap_conf)
        rank += 1
    return global_ctrl_bootstrap_confs


def _DefaultEnvProto():
    env_proto = env_pb.EnvProto()
    machine = env_proto.machine.add()
    machine.id = 0
    machine.addr = "127.0.0.1"
    return env_proto


# copied from
# https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
def _FindFreePort():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def GetEnvDefaultParallelConf(device_tag):
    if device_tag not in device_tag2default_parallel_conf:
        parallel_conf = placement_ctx.MakeParallelConf4Resource(
            device_tag, c_api_util.EnvResource()
        )
        device_tag2default_parallel_conf[device_tag] = parallel_conf
    return device_tag2default_parallel_conf[device_tag]


device_tag2default_parallel_conf = {}

default_env_proto = _DefaultEnvProto()

config_master_addr = ctrl_bootstrap_pb.Address()

config_world_size = 0

config_bootstrap_ctrl_port = 0

global_ctrl_bootstrap_confs = []
