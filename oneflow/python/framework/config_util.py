from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.core.job.resource_pb2 as resource_util

def placement(device_names):
    return placement_util.PlacementScope(placement_util.MakeParallelConf(device_names))

class JobSetConfiger(object):
    def __init__(self, job_set):
        self.job_set_ = job_set

    def gpu_device_num(self, val):
        assert type(val) is int
        self.job_set_.resource.gpu_device_num = val
        return self

    def cpu_device_num(self, val):
        assert type(val) is int
        self.job_set_.resource.cpu_device_num = val
        return self

    def machine(self, val):
        self.job_set_.resource.machine.extend(_MakeMachine(val))
        return self
    
    def ctrl_port(self, val):
        assert type(val) is int
        self.job_set_.resource.ctrl_port = val
        return self
    
    def data_port(self, val):
        assert type(val) is int
        self.job_set_.resource.data_port = val
        return self

    def comm_net_worker_num(self, val):
        assert type(val) is int
        self.job_set_.resource.comm_net_worker_num = val
        return self

    def max_mdsave_worker_num(self, val):
        assert type(val) is int
        self.job_set_.resource.max_mdsave_worker_num = val
        return self

    def rdma_mem_block_mbyte(self, val):
        assert type(val) is int
        self.job_set_.resource.rdma_mem_block_mbyte = val
        return self

    def rdma_recv_msg_buf_mbyte(self, val):
        assert type(val) is int
        self.job_set_.resource.rdma_recv_msg_buf_mbyte = val
        return self

    def reserved_host_mem_mbyte(self, val):
        assert type(val) is int
        self.job_set_.resource.reserved_host_mem_mbyte = val
        return self

    def reserved_device_mem_mbyte(self, val):
        assert type(val) is int
        self.job_set_.resource.reserved_device_mem_mbyte = val
        return self

    def use_rdma(self, val = True):
        assert type(val) is bool
        self.job_set_.resource.use_rdma = val
        return self

    def model_load_snapshot_path(self, val):
        assert type(val) is str
        self.job_set_.io_conf.model_load_snapshot_path = val
        return self

    def model_save_snapshots_path(self, val):
        assert type(val) is str
        self.job_set_.io_conf.model_save_snapshots_path = val
        return self

    def enable_write_snapshot(self, val = True):
        assert type(val) is bool
        self.job_set_.io_conf.enable_write_snapshot = val
        return self

    def save_downloaded_file_to_local_fs(self, val = True):
        assert type(val) is bool
        self.job_set_.io_conf.save_downloaded_file_to_local_fs = val
        return self

    def persistence_buf_byte(self, val):
        assert type(val) is int
        self.job_set_.io_conf.persistence_buf_byte = val
        return self

    def log_dir(self, val):
        assert type(val) is str
        self.job_set_.cpp_flags_conf.log_dir = val
        return self

    def logtostderr(self, val):
        assert type(val) is int
        self.job_set_.cpp_flags_conf.logtostderr = val
        return self

    def logbuflevel(self, val):
        assert type(val) is int
        self.job_set_.cpp_flags_conf.logbuflevel = val
        return self

    def v(self, val):
        assert type(val) is int
        self.job_set_.cpp_flags_conf.v = val
        return self

    def grpc_use_no_signal(self, val = True):
        assert type(val) is bool
        self.job_set_.cpp_flags_conf.grpc_use_no_signal = val
        return self

    def collect_act_event(self, val = True):
        assert type(val) is int
        self.job_set_.profile_conf.collect_act_event = val
        return self

class JobConfiger(object):
    def __init__(self, job_conf):
        self.job_conf_ = job_conf

    def batch_size(self, val):
        assert type(val) is int
        self.job_conf_.other.piece_size = val # it's not a type
        return self
    
    def default_data_type(self, val):
        assert type(val) is int
        self.job_conf_.other.default_data_type = val
        return self

    def data_part_num(self, val):
        assert type(val) is int
        self.job_conf_.other.data_part_num = val
        return self

    def enable_cudnn(self, val = True):
        assert type(val) is bool
        self.job_conf_.other.enable_cudnn = val
        return self

    def cudnn_buf_limit_mbyte(self, val):
        assert type(val) is int
        self.job_conf_.other.cudnn_buf_limit_mbyte = val
        return self

    def enable_mem_sharing(self, val = True):
        assert type(val) is bool
        self.job_conf_.other.enable_mem_sharing = val
        return self

    def enable_inplace(self, val = True):
        assert type(val) is bool
        self.job_conf_.other.enable_inplace = val
        return self

    def enable_nccl(self, val = True):
        assert type(val) is bool
        self.job_conf_.other.enable_nccl = val
        return self

    def use_nccl_inter_node_communication(self, val = True):
        assert type(val) is bool
        self.job_conf_.other.use_nccl_inter_node_communication = val
        return self

    def all_reduce_group_num(self, val):
        assert type(val) is int
        self.job_conf_.other.all_reduce_group_num = val
        return self

    def all_reduce_lazy_ratio(self, val):
        assert type(val) is float
        self.job_conf_.other.all_reduce_lazy_ratio = val
        return self

    def all_reduce_group_min_mbyte(self, val):
        assert type(val) is int
        self.job_conf_.other.all_reduce_group_min_mbyte = val
        return self

    def all_reduce_group_size_warmup(self, val):
        assert type(val) is float
        self.job_conf_.other.all_reduce_group_size_warmup = val
        return self

    def all_reduce_fp16(self, val = True):
        assert type(val) is bool
        self.job_conf_.other.all_reduce_fp16 = val
        return self

    def concurrency_width(self, val):
        assert type(val) is int
        self.job_conf_.other.concurrency_width = val
        return self

    def train(self):
        # self.job_conf_.other.train_conf.SetInParent()
        # return self.self.job_conf_.other.train_conf
        self.job_conf_.other.predict.tmp_split_fw_bw_train_conf.SetInParent()
        return self.self.job_conf_.other.tmp_split_fw_bw_train_conf

def DefaultConfigJobSet(job_set):
    _DefaultConfigResource(job_set)
    _DefaultConfigIO(job_set)
    _DefaultConfigCppFlags(job_set)

def DefaultConfigJobConf(job_conf):
    _DefaultConfigJobConf(job_conf)

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
    
def _DefaultConfigResource(job_set):
    resource = job_set.resource
    if len(resource.machine) == 0:
        machine = resource.machine.add()
        machine.id = 0
        machine.addr = "127.0.0.1"
    if resource.HasField("ctrl_port") == False:
        resource.ctrl_port = 2017

def _DefaultConfigIO(job_set):
    io_conf = job_set.io_conf
    if io_conf.data_fs_conf.WhichOneof("fs_type") == None:
        io_conf.data_fs_conf.localfs_conf.SetInParent()
    if io_conf.snapshot_fs_conf.WhichOneof("fs_type") == None:
        io_conf.snapshot_fs_conf.localfs_conf.SetInParent()
        
def  _DefaultConfigCppFlags(job_set):
    job_set.cpp_flags_conf.SetInParent()

    
def _DefaultConfigJobConf(job_conf):
    assert job_conf.other.HasField('piece_size'), "batch_size unset"
    other = job_conf.other
    if other.WhichOneof("job_type") is None:
        other.predict_conf.SetInParent()
    if other.HasField('train_conf'):
        other.train_conf.batch_size = other.piece_size
    if other.predict_conf.HasField('tmp_split_fw_bw_train_conf'):
        other.predict_conf.tmp_split_fw_bw_train_conf.batch_size = other.piece_size
