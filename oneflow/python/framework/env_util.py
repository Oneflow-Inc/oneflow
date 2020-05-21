from __future__ import absolute_import

import socket
from contextlib import closing
import oneflow.core.job.env_pb2 as env_pb
import oneflow.python.framework.hob as hob
import oneflow.python.framework.c_api_util as c_api_util
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('env.init', enable_if=hob.in_normal_mode & hob.env_initialized)
def env_init():
    print("Nothing happened because environment has been initialized")
    return False

@oneflow_export('env.init', enable_if=hob.in_normal_mode & ~hob.env_initialized)
def env_init():
    global default_env_proto
    assert len(default_env_proto.machine) > 0
    CompleteEnvProto(default_env_proto)
    c_api_util.InitEnv(default_env_proto)
    global env_proto_mutable
    env_proto_mutable = False
    return True

@oneflow_export('env.machine')
def machine(*val):
    assert env_proto_mutable == True
    del default_env_proto.machine[:]
    if len(val) == 1 and isinstance(val[0], (list, tuple)): val = val[0]
    default_env_proto.ClearField('machine')
    default_env_proto.machine.extend(_MakeMachine(val))

@oneflow_export('env.ctrl_port')
def ctrl_port(val):
    assert env_proto_mutable == True
    assert type(val) is int
    default_env_proto.ctrl_port = val

@oneflow_export('env.data_port')
def data_port(val):
    assert env_proto_mutable == True
    assert type(val) is int
    default_env_proto.data_port = val

@oneflow_export('env.grpc_use_no_signal')
def grpc_use_no_signal(val = True):
    assert env_proto_mutable == True
    assert type(val) is bool
    default_env_proto.grpc_use_no_signal = val

@oneflow_export('env.log_dir')
def log_dir(val):
    assert env_proto_mutable == True
    assert type(val) is str
    default_env_proto.cpp_logging_conf.log_dir = val

@oneflow_export('env.logtostderr')
def logtostderr(val):
    assert env_proto_mutable == True
    assert type(val) is int
    default_env_proto.cpp_logging_conf.logtostderr = val

@oneflow_export('env.logbuflevel')
def logbuflevel(val):
    assert env_proto_mutable == True
    assert type(val) is int
    default_env_proto.cpp_logging_conf.logbuflevel = val

def _MakeMachine(machines):
    if isinstance(machines, str): machines = [machines]
    rp_machine = env_pb.EnvProto().machine
    for m_data in machines:
        m = rp_machine.add()
        if isinstance(m_data, str):
            m.addr = m_data
        elif isinstance(m_data, dict):
            if 'addr' in m_data: m.addr = m_data['addr']
            if 'ctrl_port_agent' in m_data: m.ctrl_port_agent = m_data['ctrl_port_agent']
            if 'data_port_agent' in m_data: m.data_port_agent = m_data['data_port_agent']
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

def CompleteEnvProto(env_proto):
    if len(env_proto.machine) == 1 and env_proto.HasField('ctrl_port') == False:
        env_proto.ctrl_port = _FindFreePort()

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
        s.bind(('localhost', 0)) 
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

default_env_proto = _DefaultEnvProto()
env_proto_mutable = True
