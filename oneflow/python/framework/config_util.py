from __future__ import absolute_import

import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.compile_context as compile_context
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.pb_util as pb_util

def TryCompleteDefaultJobConfigProto(job_conf):
    _TryCompleteDefaultJobConfigProto(job_conf)

def _TryCompleteDefaultConfigProto(config):
    _DefaultConfigResource(config)
    _DefaultConfigIO(config)
    _DefaultConfigCppFlags(config)

def _MakeMachine(machines):
    if isinstance(machines, str): machines = [machines]
    resource = resource_util.Resource()
    rp_machine = resource.machine
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

def _DefaultConfigResource(config):
    resource = config.resource
    if len(resource.machine) == 0:
        machine = resource.machine.add()
        machine.id = 0
        machine.addr = "127.0.0.1"
    if resource.HasField("ctrl_port") == False:
        resource.ctrl_port = 2017
    if resource.gpu_device_num == 0:
        resource.gpu_device_num = 1

def _DefaultConfigIO(config):
    io_conf = config.io_conf
    if io_conf.data_fs_conf.WhichOneof("fs_type") == None:
        io_conf.data_fs_conf.localfs_conf.SetInParent()
    if io_conf.snapshot_fs_conf.WhichOneof("fs_type") == None:
        io_conf.snapshot_fs_conf.localfs_conf.SetInParent()

def  _DefaultConfigCppFlags(config):
    config.cpp_flags_conf.SetInParent()
    config.cpp_flags_conf.grpc_use_no_signal = True


def _TryCompleteDefaultJobConfigProto(job_conf):
    if job_conf.WhichOneof("job_type") is None:
        job_conf.predict_conf.SetInParent()

def _DefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    _TryCompleteDefaultConfigProto(config_proto)
    return config_proto

config_proto = _DefaultConfigProto()
config_proto_mutable = True

@oneflow_export('config.gpu_device_num')
def gpu_device_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.gpu_device_num = val

@oneflow_export('config.cpu_device_num')
def cpu_device_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.cpu_device_num = val

@oneflow_export('config.machine')
def machine(val):
    assert config_proto_mutable == True
    del config_proto.resource.machine[:]
    config_proto.resource.machine.extend(_MakeMachine(val))

@oneflow_export('config.ctrl_port')
def ctrl_port(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.ctrl_port = val

@oneflow_export('config.data_port')
def data_port(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.data_port = val

@oneflow_export('config.comm_net_worker_num')
def comm_net_worker_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.comm_net_worker_num = val

@oneflow_export('config.max_mdsave_worker_num')
def max_mdsave_worker_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.max_mdsave_worker_num = val

@oneflow_export('config.rdma_mem_block_mbyte')
def rdma_mem_block_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.rdma_mem_block_mbyte = val

@oneflow_export('config.rdma_recv_msg_buf_mbyte')
def rdma_recv_msg_buf_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.rdma_recv_msg_buf_mbyte = val

@oneflow_export('config.reserved_host_mem_mbyte')
def reserved_host_mem_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.reserved_host_mem_mbyte = val

@oneflow_export('config.reserved_device_mem_mbyte')
def reserved_device_mem_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.resource.reserved_device_mem_mbyte = val

@oneflow_export('config.use_rdma')
def use_rdma(val = True):
    assert config_proto_mutable == True
    assert type(val) is bool
    config_proto.resource.use_rdma = val

@oneflow_export('config.save_downloaded_file_to_local_fs')
def save_downloaded_file_to_local_fs(val = True):
    assert config_proto_mutable == True
    assert type(val) is bool
    config_proto.io_conf.save_downloaded_file_to_local_fs = val

@oneflow_export('config.persistence_buf_byte')
def persistence_buf_byte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.io_conf.persistence_buf_byte = val

@oneflow_export('config.log_dir')
def log_dir(val):
    assert config_proto_mutable == True
    assert type(val) is str
    config_proto.cpp_flags_conf.log_dir = val

@oneflow_export('config.logtostderr')
def logtostderr(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.cpp_flags_conf.logtostderr = val

@oneflow_export('config.logbuflevel')
def logbuflevel(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.cpp_flags_conf.logbuflevel = val

@oneflow_export('config.v')
def v(val):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.cpp_flags_conf.v = val

@oneflow_export('config.grpc_use_no_signal')
def grpc_use_no_signal(val = True):
    assert config_proto_mutable == True
    assert type(val) is bool
    config_proto.cpp_flags_conf.grpc_use_no_signal = val

@oneflow_export('config.collect_act_event')
def collect_act_event(val = True):
    assert config_proto_mutable == True
    assert type(val) is int
    config_proto.profile_conf.collect_act_event = val

class ConfigTrainConfBuilder(object):
    def __init__(self, train_conf):
        self.train_conf_ = train_conf

    def add_loss(self, *args):
        loss_blob_list = []
        assert(len(args) > 0)
        if type(args[0]) in [list, tuple]:
            assert(len(args) == 1)
            loss_blob_list = args[0]
        else:
            loss_blob_list = args
        self.train_conf_.loss_lbn.extend(
            [b.logical_blob_name for b in loss_blob_list])

class JobConfigProtoBuilder(object):
    def __init__(self, job_conf):
        assert isinstance(job_conf, job_util.JobConfigProto)
        self.job_conf_ = job_conf

    def job_conf():
        return self.job_conf_

    def default_initializer_conf(self, val):
        assert type(val) is dict
        pb_util.PythonDict2PbMessage(val, self.job_conf_.default_initializer_conf)
        return self

    def model_update_conf(self, val):
        assert type(val) is dict
        assert self.job_conf_.HasField("train_conf")
        pb_util.PythonDict2PbMessage(val, self.train_conf().model_update_conf)
        return self

    def batch_size(self, val):
        assert type(val) is int
        self.job_conf_.piece_size = val # it's not a type
        return self

    def default_data_type(self, val):
        assert type(val) is int
        self.job_conf_.default_data_type = val
        return self

    def data_part_num(self, val):
        assert type(val) is int
        self.job_conf_.data_part_num = val
        return self

    def enable_cudnn(self, val = True):
        assert type(val) is bool
        self.job_conf_.enable_cudnn = val
        return self

    def cudnn_buf_limit_mbyte(self, val):
        assert type(val) is int
        self.job_conf_.cudnn_buf_limit_mbyte = val
        return self

    def enable_mem_sharing(self, val = True):
        assert type(val) is bool
        self.job_conf_.enable_mem_sharing = val
        return self

    def enable_inplace(self, val = True):
        assert type(val) is bool
        self.job_conf_.enable_inplace = val
        return self

    def enable_nccl(self, val = True):
        assert type(val) is bool
        self.job_conf_.enable_nccl = val
        return self

    def use_nccl_inter_node_communication(self, val = True):
        assert type(val) is bool
        self.job_conf_.use_nccl_inter_node_communication = val
        return self
    
    def enable_all_reduce_group(self, val = True):
        assert type(val) is bool
        self.job_conf_.enable_all_reduce_group = val
        return self

    def all_reduce_group_num(self, val):
        assert type(val) is int
        self.job_conf_.all_reduce_group_num = val
        return self

    def all_reduce_lazy_ratio(self, val):
        assert type(val) is float
        self.job_conf_.all_reduce_lazy_ratio = val
        return self

    def all_reduce_group_min_mbyte(self, val):
        assert type(val) is int
        self.job_conf_.all_reduce_group_min_mbyte = val
        return self

    def all_reduce_group_size_warmup(self, val):
        assert type(val) is float
        self.job_conf_.all_reduce_group_size_warmup = val
        return self

    def all_reduce_fp16(self, val = True):
        assert type(val) is bool
        self.job_conf_.all_reduce_fp16 = val
        return self

    def concurrency_width(self, val):
        assert type(val) is int
        self.job_conf_.concurrency_width = val
        return self

    def train_conf(self):
        self.job_conf_.train_conf.SetInParent()
        return self.job_conf_.train_conf

    def get_train_conf_builder(self):
        return ConfigTrainConfBuilder(self.train_conf())
