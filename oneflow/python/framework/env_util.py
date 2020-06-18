from __future__ import absolute_import

import socket
from contextlib import closing

import oneflow.core.job.env_pb2 as env_pb
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("env.init")
def api_env_init():
    return enable_if.unique(env_init, do_nothing)()


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def env_init():
    global default_env_proto
    assert len(default_env_proto.machine) > 0
    CompleteEnvProto(default_env_proto)
    c_api_util.InitEnv(default_env_proto)
    global env_proto_mutable
    env_proto_mutable = False
    return True


@oneflow_export("env.current_resource", "current_resource")
def api_get_current_resource():
    return enable_if.unique(get_current_resource)()


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def get_current_resource():
    return c_api_util.CurrentResource()


@oneflow_export("current_machine_id")
def api_get_current_machine_id():
    return enable_if.unique(get_current_machine_id)()


@enable_if.condition(hob.in_normal_mode & hob.env_initialized)
def get_current_machine_id():
    return c_api_util.CurrentMachineId()


@oneflow_export("env.machine")
def api_machine(*val):
    r"""Set machines' hostnames.  For instance::

        oneflow.env.machine([{"addr": "192.168.1.1"}, {"addr": "192.168.1.2"}])

    Args:
        val:  `list`, `tuple` or multiple arguments of `dict`. First in the list is the master machine.
    """
    return enable_if.unique(machine, do_nothing)(*val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def machine(*val):

    del default_env_proto.machine[:]
    if len(val) == 1 and isinstance(val[0], (list, tuple)):
        val = val[0]
    default_env_proto.ClearField("machine")
    default_env_proto.machine.extend(_MakeMachine(val))


@oneflow_export("env.ctrl_port")
def api_ctrl_port(val):
    r"""Set port number used to control the execution across multiple machines. Same on every machine.

    Args:
        val: a port number accessible to peer machines
    """
    return enable_if.unique(ctrl_port, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def ctrl_port(val):
    assert type(val) is int
    default_env_proto.ctrl_port = val


@oneflow_export("env.data_port")
def api_data_port(val):
    r"""Set port number used to data transfer among multiple machines. Same on every machine.

    Args:
        val: a port number accessible to peer machines
    """
    return enable_if.unique(data_port, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def data_port(val):
    assert type(val) is int
    default_env_proto.data_port = val


@oneflow_export("env.grpc_use_no_signal")
def api_grpc_use_no_signal(val=True):
    return enable_if.unique(grpc_use_no_signal, do_nothing)(val=True)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def grpc_use_no_signal(val=True):
    assert type(val) is bool
    default_env_proto.grpc_use_no_signal = val


@oneflow_export("env.log_dir")
def api_log_dir(val):
    r"""Specify a dir to store OneFlow's logging files. If not specified, it is `./log` by default.

    """
    return enable_if.unique(log_dir, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def log_dir(val):
    assert type(val) is str
    default_env_proto.cpp_logging_conf.log_dir = val


@oneflow_export("env.logtostderr")
def api_logtostderr(val):
    return enable_if.unique(logtostderr, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.env_initialized)
def logtostderr(val):
    assert type(val) is int
    default_env_proto.cpp_logging_conf.logtostderr = val


@oneflow_export("env.logbuflevel")
def api_logbuflevel(val):
    return enable_if.unique(logbuflevel, do_nothing)(val)


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


def _DefaultEnvProto():
    env_proto = env_pb.EnvProto()
    machine = env_proto.machine.add()
    machine.id = 0
    machine.addr = "127.0.0.1"
    env_proto.grpc_use_no_signal = True
    return env_proto


# copied from
# https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
def _FindFreePort():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


default_env_proto = _DefaultEnvProto()
env_proto_mutable = True
