from __future__ import absolute_import

import oneflow.core.job.cluster_pb2 as cluster_pb
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('env.machine')
def machine(*val):
    assert cluster_proto_mutable == True
    del default_cluster_proto.machine[:]
    if len(val) == 1 and isinstance(val[0], (list, tuple)): val = val[0]
    default_cluster_proto.ClearField('machine')
    default_cluster_proto.machine.extend(_MakeMachine(val))

@oneflow_export('env.ctrl_port')
def ctrl_port(val):
    assert cluster_proto_mutable == True
    assert type(val) is int
    default_cluster_proto.ctrl_port = val

@oneflow_export('env.data_port')
def data_port(val):
    assert cluster_proto_mutable == True
    assert type(val) is int
    default_cluster_proto.data_port = val

@oneflow_export('env.grpc_use_no_signal')
def grpc_use_no_signal(val = True):
    assert cluster_proto_mutable == True
    assert type(val) is bool
    default_cluster_proto.grpc_use_no_signal = val

@oneflow_export('env.log_dir')
def log_dir(val):
    assert config_proto_mutable == True
    assert type(val) is str
    default_config_proto.cpp_logging_conf.log_dir = val

@oneflow_export('env.logtostderr')
def logtostderr(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.cpp_logging_conf.logtostderr = val

@oneflow_export('env.logbuflevel')
def logbuflevel(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.cpp_logging_conf.logbuflevel = val

def _MakeMachine(machines):
    if isinstance(machines, str): machines = [machines]
    rp_machine = cluster_pb.ClusterProto().machine
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

def _DefaultClusterProto():
    cluster_proto = cluster_pb.ClusterProto()
    machine = cluster_proto.machine.add()
    machine.id = 0
    machine.addr = "127.0.0.1"
    cluster_proto.ctrl_port = 2017
    cluster_proto.grpc_use_no_signal = True
    return cluster_proto

default_cluster_proto = _DefaultClusterProto()
cluster_proto_mutable = True
