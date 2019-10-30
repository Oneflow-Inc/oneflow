from __future__ import absolute_import

import oneflow
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.job_pb2 as job_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.c_api_util as c_api_util
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

default_config_proto = _DefaultConfigProto()
config_proto_mutable = True

@oneflow_export('config.gpu_device_num')
def gpu_device_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.gpu_device_num = val

@oneflow_export('config.cpu_device_num')
def cpu_device_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.cpu_device_num = val

@oneflow_export('config.machine')
def machine(val):
    assert config_proto_mutable == True
    del default_config_proto.resource.machine[:]
    default_config_proto.resource.machine.extend(_MakeMachine(val))

@oneflow_export('config.ctrl_port')
def ctrl_port(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.ctrl_port = val

@oneflow_export('config.data_port')
def data_port(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.data_port = val

@oneflow_export('config.comm_net_worker_num')
def comm_net_worker_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.comm_net_worker_num = val

@oneflow_export('config.max_mdsave_worker_num')
def max_mdsave_worker_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.max_mdsave_worker_num = val

@oneflow_export('config.compute_thread_pool_size')
def max_mdsave_worker_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.compute_thread_pool_size = val

@oneflow_export('config.rdma_mem_block_mbyte')
def rdma_mem_block_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.rdma_mem_block_mbyte = val

@oneflow_export('config.rdma_recv_msg_buf_mbyte')
def rdma_recv_msg_buf_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.rdma_recv_msg_buf_mbyte = val

@oneflow_export('config.reserved_host_mem_mbyte')
def reserved_host_mem_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.reserved_host_mem_mbyte = val

@oneflow_export('config.reserved_device_mem_mbyte')
def reserved_device_mem_mbyte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.reserved_device_mem_mbyte = val

@oneflow_export('config.use_rdma')
def use_rdma(val = True):
    assert config_proto_mutable == True
    assert type(val) is bool
    default_config_proto.resource.use_rdma = val

@oneflow_export('config.save_downloaded_file_to_local_fs')
def save_downloaded_file_to_local_fs(val = True):
    assert config_proto_mutable == True
    assert type(val) is bool
    default_config_proto.io_conf.save_downloaded_file_to_local_fs = val

@oneflow_export('config.persistence_buf_byte')
def persistence_buf_byte(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.io_conf.persistence_buf_byte = val

@oneflow_export('config.enable_model_io_v2')
def enable_model_io_v2(val):
    assert config_proto_mutable == True
    assert type(val) is bool
    default_config_proto.io_conf.enable_model_io_v2 = val

@oneflow_export('config.log_dir')
def log_dir(val):
    assert config_proto_mutable == True
    assert type(val) is str
    default_config_proto.cpp_flags_conf.log_dir = val

@oneflow_export('config.logtostderr')
def logtostderr(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.cpp_flags_conf.logtostderr = val

@oneflow_export('config.logbuflevel')
def logbuflevel(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.cpp_flags_conf.logbuflevel = val

@oneflow_export('config.v')
def v(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.cpp_flags_conf.v = val

@oneflow_export('config.grpc_use_no_signal')
def grpc_use_no_signal(val = True):
    assert config_proto_mutable == True
    assert type(val) is bool
    default_config_proto.cpp_flags_conf.grpc_use_no_signal = val

@oneflow_export('config.collect_act_event')
def collect_act_event(val = True):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.profile_conf.collect_act_event = val

@oneflow_export('config.total_batch_num')
def set_total_batch_num(value):
    _SetJobConfAttr(lambda x:x, 'total_batch_num', value)
    return oneflow.config

@oneflow_export('config.default_data_type')
def set_default_data_type(value):
    _SetJobConfAttr(lambda x:x, 'default_data_type', value)
    return oneflow.config

@oneflow_export('config.max_data_id_length')
def set_max_data_id_length(value):
    _SetJobConfAttr(lambda x:x, 'max_data_id_length', value)
    return oneflow.config

@oneflow_export('config.default_initializer_conf')
def set_default_initializer_conf(value):
    assert type(value) is dict
    pb_msg = _GetJobConfAttr(lambda x:x, 'default_initializer_conf')
    pb_util.PythonDict2PbMessage(value, pb_msg)
    return oneflow.config

@oneflow_export('config.exp_run_conf')
def set_exp_run_conf(value):
    assert type(value) is dict
    pb_util.PythonDict2PbMessage(value, _GetJobConfAttr(lambda x:x, 'exp_run_conf'))
    return oneflow.config

@oneflow_export('config.use_memory_allocation_algorithm_v2')
def set_use_memory_allocation_algorithm_v2(value = True):
    _SetJobConfAttr(lambda x:x, 'use_memory_allocation_algorithm_v2', value)
    return oneflow.config

@oneflow_export('config.static_mem_alloc_algo_white_list.show')
def show_static_mem_alloc_algo_white_list():
    return ["use_mem_size_first_algo", "use_mutual_exclusion_first_algo", "use_time_line_algo"]

@oneflow_export('config.static_mem_alloc_policy_white_list.has')
def static_mem_alloc_policy_white_list_has_policy(policy):
    pb_msg = _GetJobConfAttr(lambda x:x, 'memory_allocation_algorithm_conf')
    return getattr(pb_msg, policy)

@oneflow_export('config.static_mem_alloc_policy_white_list.add')
def static_mem_alloc_policy_white_list_add_policy(policy):
    pb_msg = _GetJobConfAttr(lambda x:x, 'memory_allocation_algorithm_conf')
    setattr(pb_msg, policy, True)
    return oneflow.config

@oneflow_export('config.static_mem_alloc_policy_white_list.remove')
def static_mem_alloc_policy_white_list_remove_policy(policy):
    pb_msg = _GetJobConfAttr(lambda x:x, 'memory_allocation_algorithm_conf')
    setattr(pb_msg, policy, False)
    return oneflow.config

@oneflow_export('config.static_mem_alloc_policy_white_list.policy_mem_size_first')
def policy_mem_size_first():
    return "use_mem_size_first_algo"

@oneflow_export('config.static_mem_alloc_policy_white_list.policy_mutual_exclusion_first')
def policy_mutual_exclusion_first():
    return "use_mutual_exclusion_first_algo"

@oneflow_export('config.static_mem_alloc_policy_white_list.policy_time_line')
def policy_time_line():
    return "use_time_line_algo"

@oneflow_export('config.enable_cudnn')
def set_enable_cudnn(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_cudnn', value)
    return oneflow.config

@oneflow_export('config.cudnn_buf_limit_mbyte')
def set_cudnn_buf_limit_mbyte(value):
    _SetJobConfAttr(lambda x:x, 'cudnn_buf_limit_mbyte', value)
    return oneflow.config

@oneflow_export('config.cudnn_conv_force_fwd_algo')
def set_cudnn_conv_force_fwd_algo(value):
    _SetJobConfAttr(lambda x:x, 'cudnn_conv_force_fwd_algo', value)
    return oneflow.config

@oneflow_export('config.cudnn_conv_force_bwd_data_algo')
def set_cudnn_conv_force_bwd_data_algo(value):
    _SetJobConfAttr(lambda x:x, 'cudnn_conv_force_bwd_data_algo', value)
    return oneflow.config

@oneflow_export('config.cudnn_conv_force_bwd_filter_algo')
def set_cudnn_conv_force_bwd_filter_algo(value):
    _SetJobConfAttr(lambda x:x, 'cudnn_conv_force_bwd_filter_algo', value)
    return oneflow.config

@oneflow_export('config.enable_reused_mem')
def set_enable_reused_mem(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_reused_mem', value)
    return oneflow.config

@oneflow_export('config.enable_inplace')
def set_enable_inplace(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_inplace', value)
    return oneflow.config

@oneflow_export('config.enable_inplace_in_reduce_struct')
def set_enable_inplace_in_reduce_struct(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_inplace_in_reduce_struct', value)
    return oneflow.config

@oneflow_export('config.enable_nccl')
def set_enable_nccl(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_nccl', value)
    return oneflow.config

@oneflow_export('config.use_nccl_inter_node_communication')
def set_use_nccl_inter_node_communication(value = True):
    _SetJobConfAttr(lambda x:x, 'use_nccl_inter_node_communication', value)
    return oneflow.config

@oneflow_export('config.use_boxing_v2')
def use_boxing_v2(value=True):
    _SetJobConfAttr(lambda x: x, 'use_boxing_v2', value)
    return oneflow.config

@oneflow_export('config.enable_all_reduce_group')
def set_enable_all_reduce_group(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_all_reduce_group', value)
    return oneflow.config

@oneflow_export('config.all_reduce_group_num')
def set_all_reduce_group_num(value):
    _SetJobConfAttr(lambda x:x, 'all_reduce_group_num', value)
    return oneflow.config

@oneflow_export('config.all_reduce_lazy_ratio')
def set_all_reduce_lazy_ratio(value):
    _SetJobConfAttr(lambda x:x, 'all_reduce_lazy_ratio', value)
    return oneflow.config

@oneflow_export('config.all_reduce_group_min_mbyte')
def set_all_reduce_group_min_mbyte(value):
    _SetJobConfAttr(lambda x:x, 'all_reduce_group_min_mbyte', value)
    return oneflow.config

@oneflow_export('config.all_reduce_group_size_warmup')
def set_all_reduce_group_size_warmup(value):
    _SetJobConfAttr(lambda x:x, 'all_reduce_group_size_warmup', value)
    return oneflow.config

@oneflow_export('config.all_reduce_fp16')
def set_all_reduce_fp16(value = True):
    _SetJobConfAttr(lambda x:x, 'all_reduce_fp16', value)
    return oneflow.config

@oneflow_export('config.enable_non_distributed_optimizer')
def set_enable_non_distributed_optimizer(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_non_distributed_optimizer', value)
    return oneflow.config

@oneflow_export('config.disable_all_reduce_sequence')
def disable_all_reduce_sequence(value=True):
    _SetJobConfAttr(lambda x: x, 'disable_all_reduce_sequence', value)
    return oneflow.config

@oneflow_export('config.non_distributed_optimizer_group_size_mbyte')
def set_non_distributed_optimizer_group_size_mbyte(value):
    _SetJobConfAttr(lambda x:x, 'non_distributed_optimizer_group_size_mbyte', value)
    return oneflow.config

@oneflow_export('config.enable_true_half_config_when_conv')
def set_enable_true_half_config_when_conv(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_true_half_config_when_conv', value)
    return oneflow.config

@oneflow_export('config.enable_float_compute_for_half_gemm')
def set_enable_float_compute_for_half_gemm(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_float_compute_for_half_gemm', value)
    return oneflow.config

@oneflow_export('config.enable_auto_mixed_precision')
def set_enable_auto_mixed_precision(value = True):
    _SetJobConfAttr(lambda x:x, 'enable_auto_mixed_precision', value)
    return oneflow.config

@oneflow_export('config.concurrency_width')
def set_concurrency_width(value):
    _SetJobConfAttr(lambda x:x, 'concurrency_width', value)
    return oneflow.config

@oneflow_export('config.train.model_update_conf')
def set_model_update_conf(value):
    assert type(value) is dict
    pb_msg = _GetJobConfAttr(lambda job_conf: job_conf.train_conf, 'model_update_conf')
    pb_util.PythonDict2PbMessage(value, pb_msg)
    return oneflow.config

@oneflow_export('config.train.loss_scale_factor')
def set_loss_scale_factor(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'loss_scale_factor', value)
    return oneflow.config

@oneflow_export('config.train.primary_lr')
def set_primary_lr(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'primary_lr', value)
    return oneflow.config

@oneflow_export('config.train.secondary_lr')
def set_secondary_lr(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'secondary_lr', value)
    return oneflow.config

@oneflow_export('config.train.weight_l1')
def set_weight_l1(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'weight_l1', value)
    return oneflow.config

@oneflow_export('config.train.bias_l1')
def set_bias_l1(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'bias_l1', value)
    return oneflow.config

@oneflow_export('config.train.weight_l2')
def set_weight_l2(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'weight_l2', value)
    return oneflow.config

@oneflow_export('config.train.bias_l2')
def set_bias_l2(value):
    _SetJobConfAttr(lambda job_conf: job_conf.train_conf, 'bias_l2', value)
    return oneflow.config

def _SetJobConfAttr(GetConf, field, value):
    if compile_context.cur_job_conf is not None:
        assert c_api_util.CurJobBuildAndInferCtx_HasJobConf() == False
        setattr(GetConf(compile_context.cur_job_conf), field, value)
    else:
        assert c_api_util.IsEnvironmentInited() == False
        setattr(GetConf(default_job_conf), field, value)

def _GetJobConfAttr(GetConf, field):
    if compile_context.cur_job_conf is not None:
        assert c_api_util.CurJobBuildAndInferCtx_HasJobConf() == False
        return getattr(GetConf(compile_context.cur_job_conf), field)
    else:
        assert c_api_util.IsEnvironmentInited() == False
        return getattr(GetConf(default_job_conf), field)

default_job_conf = job_util.JobConfigProto()
