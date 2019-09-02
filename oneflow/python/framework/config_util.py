from __future__ import absolute_import

import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.compile_context as compile_context
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.pb_util as pb_util

inited_config_proto = None

@oneflow_export('ConfigProtoBuilder')
class ConfigProtoBuilder(object):
    def __init__(self):
        self.config_proto_ = job_set_util.ConfigProto()

    @property
    def config_proto(self):
        return self.config_proto_

    def gpu_device_num(self, val):
        assert type(val) is int
        self.config_proto_.resource.gpu_device_num = val
        return self

    def cpu_device_num(self, val):
        assert type(val) is int
        self.config_proto_.resource.cpu_device_num = val
        return self

    def machine(self, val):
        self.config_proto_.resource.machine.extend(_MakeMachine(val))
        return self

    def ctrl_port(self, val):
        assert type(val) is int
        self.config_proto_.resource.ctrl_port = val
        return self

    def data_port(self, val):
        assert type(val) is int
        self.config_proto_.resource.data_port = val
        return self

    def comm_net_worker_num(self, val):
        assert type(val) is int
        self.config_proto_.resource.comm_net_worker_num = val
        return self

    def max_mdsave_worker_num(self, val):
        assert type(val) is int
        self.config_proto_.resource.max_mdsave_worker_num = val
        return self

    def rdma_mem_block_mbyte(self, val):
        assert type(val) is int
        self.config_proto_.resource.rdma_mem_block_mbyte = val
        return self

    def rdma_recv_msg_buf_mbyte(self, val):
        assert type(val) is int
        self.config_proto_.resource.rdma_recv_msg_buf_mbyte = val
        return self

    def reserved_host_mem_mbyte(self, val):
        assert type(val) is int
        self.config_proto_.resource.reserved_host_mem_mbyte = val
        return self

    def reserved_device_mem_mbyte(self, val):
        assert type(val) is int
        self.config_proto_.resource.reserved_device_mem_mbyte = val
        return self

    def use_rdma(self, val = True):
        assert type(val) is bool
        self.config_proto_.resource.use_rdma = val
        return self

    def model_load_snapshot_path(self, val):
        assert type(val) is str
        self.config_proto_.io_conf.model_load_snapshot_path = val
        return self

    def model_save_snapshots_path(self, val):
        assert type(val) is str
        self.config_proto_.io_conf.model_save_snapshots_path = val
        return self

    def enable_write_snapshot(self, val = True):
        assert type(val) is bool
        self.config_proto_.io_conf.enable_write_snapshot = val
        return self

    def save_downloaded_file_to_local_fs(self, val = True):
        assert type(val) is bool
        self.config_proto_.io_conf.save_downloaded_file_to_local_fs = val
        return self

    def persistence_buf_byte(self, val):
        assert type(val) is int
        self.config_proto_.io_conf.persistence_buf_byte = val
        return self

    def log_dir(self, val):
        assert type(val) is str
        self.config_proto_.cpp_flags_conf.log_dir = val
        return self

    def logtostderr(self, val):
        assert type(val) is int
        self.config_proto_.cpp_flags_conf.logtostderr = val
        return self

    def logbuflevel(self, val):
        assert type(val) is int
        self.config_proto_.cpp_flags_conf.logbuflevel = val
        return self

    def v(self, val):
        assert type(val) is int
        self.config_proto_.cpp_flags_conf.v = val
        return self

    def grpc_use_no_signal(self, val = True):
        assert type(val) is bool
        self.config_proto_.cpp_flags_conf.grpc_use_no_signal = val
        return self

    def collect_act_event(self, val = True):
        assert type(val) is int
        self.config_proto_.profile_conf.collect_act_event = val
        return self

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

def TryCompleteDefaultConfigProto(config):
    _DefaultConfigResource(config)
    _DefaultConfigIO(config)
    _DefaultConfigCppFlags(config)

def TryCompleteDefaultJobConfigProto(job_conf):
    _TryCompleteDefaultJobConfigProto(job_conf)

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

def _DefaultConfigIO(config):
    io_conf = config.io_conf
    if io_conf.data_fs_conf.WhichOneof("fs_type") == None:
        io_conf.data_fs_conf.localfs_conf.SetInParent()
    if io_conf.snapshot_fs_conf.WhichOneof("fs_type") == None:
        io_conf.snapshot_fs_conf.localfs_conf.SetInParent()

def  _DefaultConfigCppFlags(config):
    config.cpp_flags_conf.SetInParent()


def _TryCompleteDefaultJobConfigProto(job_conf):
    if job_conf.WhichOneof('job_type') is None:
        job_conf.predict_conf.SetInParent()
