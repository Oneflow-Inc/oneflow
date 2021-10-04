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

import oneflow._oneflow_internal
from oneflow import oneflow_deprecate
from oneflow.compatible.single_client.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.framework import (
    placement_context as placement_ctx,
)
from oneflow.compatible.single_client.framework import scope_util as scope_util
from oneflow.compatible.single_client.framework import session_context as session_ctx
from oneflow.compatible.single_client.support import enable_if as enable_if
from oneflow.core.control import ctrl_bootstrap_pb2 as ctrl_bootstrap_pb
from oneflow.core.job import env_pb2 as env_pb
from oneflow.core.job import resource_pb2 as resource_util


def api_all_device_placement(device_type: str) -> None:
    """Return a placement containing all devices of all machines under env.

    Args:
        device_type (str): cuda or cpu
    """
    return oneflow._oneflow_internal.AllDevicePlacement(device_type)


def api_enable_eager_execution(val: bool = True) -> None:
    """If True, job will execute in eager mode, else use lazy mode(static graph).

    Args:
        val (bool, optional): Whether  eager execution or not.  Defaults to True.
    """
    return enable_if.unique([enable_eager_environment])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def enable_eager_environment(val=True):
    return oneflow._oneflow_internal.EnableEagerEnvironment(val)


def api_env_init() -> bool:
    """Init environment for job

    Returns:
        bool: [description]
    """
    return enable_if.unique([_env_init_single_client, do_nothing])()


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def _env_init_single_client():
    return env_init(False)


def env_init(is_multi_client):
    global default_env_proto
    assert len(default_env_proto.machine) > 0
    CompleteEnvProto(default_env_proto, is_multi_client)
    c_api_util.InitEnv(default_env_proto, is_multi_client)
    if not is_multi_client:
        if oneflow._oneflow_internal.CurrentMachineId() == 0:
            scope_util.InitScopeStack()
        else:
            exit(0)
    return True


def api_get_current_resource() -> resource_util.Resource:
    """Get current resources, such as:machine nums, cpu/gpu device nums,
            epoch network threed num, rdma params...

    Returns:
        resource_util.Resource: [description]
    """
    return enable_if.unique([get_current_resource])()


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def get_current_resource():
    return c_api_util.CurrentResource()


def api_get_current_machine_id():
    """Get machine id of current machine/node

    Returns:
        [type]: [description]
    """
    return enable_if.unique([get_current_machine_id])()


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def get_current_machine_id() -> int:
    return oneflow._oneflow_internal.CurrentMachineId()


def api_machine(*val: list) -> None:
    """Set machines' hostnames.

    For instance:

        oneflow.compatible.single_client.env.machine([{"addr": "192.168.1.1"}, {"addr": "192.168.1.2"}])

    Args:
        val:  `list`, `tuple` or multiple arguments of `dict`. First in the list is the master machine.
    """
    return enable_if.unique([machine, do_nothing])(*val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def machine(*val):
    del default_env_proto.machine[:]
    if len(val) == 1 and isinstance(val[0], (list, tuple)):
        val = val[0]
    default_env_proto.ClearField("machine")
    default_env_proto.machine.extend(_MakeMachine(val))


def api_ctrl_port(val: int) -> None:
    """Set port number used to control the execution across multiple machines. Same on every machine.

    Args:
        val: a port number accessible to peer machines
    """
    return enable_if.unique([ctrl_port, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def ctrl_port(val):
    assert type(val) is int
    default_env_proto.ctrl_port = val


def api_data_port(val: int) -> None:
    """Set port number used to data transfer among multiple machines. Same on every machine.

    Args:
        val: a port number accessible to peer machines
    """
    return enable_if.unique([data_port, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def data_port(val):
    assert type(val) is int
    default_env_proto.data_port = val


from oneflow import oneflow_deprecate


@oneflow_deprecate()
def api_grpc_use_no_signal(val: bool = True) -> None:
    """Set rpc use signal or not (deprecate)

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    print(
        "WARNING:",
        "oneflow.compatible.single_client.env.grpc_use_no_signal is deprecated, users no longer need to set rpc use signal or not. \n",
        traceback.format_stack()[-2],
    )
    return None


def api_log_dir(val: str) -> None:
    """Specify a dir to store OneFlow's logging files. If not specified, it is `./log` by default.

    Args:
        val (str): string , log file path
    """
    return enable_if.unique([log_dir, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def log_dir(val):
    assert type(val) is str
    default_env_proto.cpp_logging_conf.log_dir = val


def api_logtostderr(val: int) -> None:
    """Set whether log messages go to stderr instead of logfiles

    Args:
        val (int): [description]
    """
    return enable_if.unique([logtostderr, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def logtostderr(val):
    assert type(val) is int
    default_env_proto.cpp_logging_conf.logtostderr = val


def api_logbuflevel(val: int) -> None:
    """Log messages at a level <= this flag are buffered.
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
    print("Environment has been initialized, this env init will do nothing.")
    return False


def CompleteEnvProto(env_proto, is_multi_client):
    if is_multi_client:
        _UpdateDefaultEnvProtoByMultiClientEnvVars(env_proto)
    if env_proto.HasField("ctrl_port") == False:
        if len(env_proto.machine) == 1:
            env_proto.ctrl_port = _FindFreePort()
        else:
            raise ValueError(
                "a ctrl_port is required if running multi-node, set it with 'oneflow.compatible.single_client.env.ctrl_port([YOUR PORT])'"
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


def api_init_bootstrap_confs(*val: list, **kargs) -> None:
    return enable_if.unique([MakeBootstrapConfs, do_nothing])(*val, **kargs)


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


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def MakeBootstrapConfs(
    node_list, master_port, world_size=0, ctrl_port=-1, node_size=-1
):
    """Set ctrl_bootstrap_conf' info.

    For instance:

        ONEFLOW_TEST_NODE_LIST=192.168.1.16,192.168.1.15 ONEFLOW_TEST_MASTER_PORT=43256
        ONEFLOW_TEST_WORLD_SIZE=2 ONEFLOW_TEST_RANK_CTRL_PORT=34527

    Args:
        val:  `list`, First in the list is the master machine.
    """
    if isinstance(node_list, str):
        node_list = [node_list]
    global global_ctrl_bootstrap_confs
    assert len(global_ctrl_bootstrap_confs) == 0, "ctrl_bootstrap_conf has been inited"
    global config_master_addr
    config_master_addr.host = node_list[0]
    config_master_addr.port = master_port
    global config_world_size
    if world_size == 0:
        config_world_size = len(node_list)
    else:
        assert world_size % len(node_list) == 0
        config_world_size = world_size
    global config_bootstrap_ctrl_port
    if ctrl_port != -1:
        config_bootstrap_ctrl_port = ctrl_port
    global config_node_size
    if node_size != -1:
        config_node_size = node_size
    rank = 0
    for rank_host in node_list:
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


def HasAllMultiClientEnvVars():
    return (
        os.getenv("MASTER_ADDR")
        and os.getenv("MASTER_PORT")
        and os.getenv("WORLD_SIZE")
        and os.getenv("RANK")
        and os.getenv("LOCAL_RANK")
    )


def _UpdateDefaultEnvProtoByMultiClientEnvVars(env_proto):
    assert HasAllMultiClientEnvVars()

    def str2int(env_config):
        assert env_config.isdigit()
        return int(env_config)

    bootstrap_conf = ctrl_bootstrap_pb.BootstrapConf()
    master_addr = ctrl_bootstrap_pb.Address()
    master_addr.host = os.getenv("MASTER_ADDR")
    master_addr.port = str2int(os.getenv("MASTER_PORT"))
    bootstrap_conf.master_addr.CopyFrom(master_addr)
    bootstrap_conf.world_size = str2int(os.getenv("WORLD_SIZE"))
    bootstrap_conf.rank = str2int(os.getenv("RANK"))
    env_proto.ctrl_bootstrap_conf.CopyFrom(bootstrap_conf)


device_tag2default_parallel_conf = {}
default_env_proto = _DefaultEnvProto()
config_master_addr = ctrl_bootstrap_pb.Address()
config_world_size = 0
config_bootstrap_ctrl_port = 0
config_node_size = 0
global_ctrl_bootstrap_confs = []
