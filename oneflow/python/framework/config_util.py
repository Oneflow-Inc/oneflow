from __future__ import absolute_import

import oneflow
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.job.job_set_pb2 as job_set_pb
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.c_api_util as c_api_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.pb_util as pb_util

def _TryCompleteDefaultConfigProto(config):
    _DefaultConfigResource(config)
    _DefaultConfigIO(config)

def _DefaultConfigResource(config):
    resource = config.resource
    if resource.gpu_device_num == 0:
        resource.gpu_device_num = 1

def _DefaultConfigIO(config):
    io_conf = config.io_conf
    if io_conf.data_fs_conf.WhichOneof("fs_type") == None:
        io_conf.data_fs_conf.localfs_conf.SetInParent()
    if io_conf.snapshot_fs_conf.WhichOneof("fs_type") == None:
        io_conf.snapshot_fs_conf.localfs_conf.SetInParent()

def _DefaultConfigProto():
    config_proto = job_set_pb.ConfigProto()
    _TryCompleteDefaultConfigProto(config_proto)
    return config_proto

@oneflow_export('config.machine_num')
def machine_num(val):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.resource.machine_num = val

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

@oneflow_export('config.thread_enable_local_message_queue')
def thread_enable_local_message_queue(val):
    assert config_proto_mutable == True
    assert type(val) is bool
    default_config_proto.resource.thread_enable_local_message_queue = val

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

@oneflow_export('config.collect_act_event')
def collect_act_event(val = True):
    assert config_proto_mutable == True
    assert type(val) is int
    default_config_proto.profile_conf.collect_act_event = val

default_config_proto = _DefaultConfigProto()
config_proto_mutable = True
