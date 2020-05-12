from __future__ import absolute_import
from __future__ import print_function

import sys
import oneflow
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.job.job_set_pb2 as job_set_pb
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.g_func_ctx as g_func_ctx
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.pb_util as pb_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.hob as hob

@oneflow_export('config.load_library', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def load_libray(val):
    assert type(val) is str
    sess = session_ctx.GetDefaultSession()
    sess.config_proto.load_lib_path.append(val)

@oneflow_export('config.machine_num', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def machine_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.machine_num = val

@oneflow_export('config.gpu_device_num', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def gpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.gpu_device_num = val

@oneflow_export('config.cpu_device_num', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def cpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.cpu_device_num = val

@oneflow_export('config.comm_net_worker_num', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def comm_net_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.comm_net_worker_num = val

@oneflow_export('config.max_mdsave_worker_num', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def max_mdsave_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.max_mdsave_worker_num = val

@oneflow_export('config.compute_thread_pool_size', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def max_mdsave_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.compute_thread_pool_size = val

@oneflow_export('config.rdma_mem_block_mbyte', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def rdma_mem_block_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.rdma_mem_block_mbyte = val

@oneflow_export('config.rdma_recv_msg_buf_mbyte', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def rdma_recv_msg_buf_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.rdma_recv_msg_buf_mbyte = val

@oneflow_export('config.reserved_host_mem_mbyte', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def reserved_host_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_host_mem_mbyte = val

@oneflow_export('config.reserved_device_mem_mbyte', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def reserved_device_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_device_mem_mbyte = val

@oneflow_export('config.use_rdma', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def use_rdma(val = True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.use_rdma = val

@oneflow_export('config.thread_enable_local_message_queue', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def thread_enable_local_message_queue(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.thread_enable_local_message_queue = val


@oneflow_export('config.enable_debug_mode', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def enable_debug_mode(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_debug_mode = val

@oneflow_export('config.save_downloaded_file_to_local_fs', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def save_downloaded_file_to_local_fs(val = True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.io_conf.save_downloaded_file_to_local_fs = val

@oneflow_export('config.persistence_buf_byte', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def persistence_buf_byte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.io_conf.persistence_buf_byte = val

@oneflow_export('config.collect_act_event', enable_if = hob.in_normal_mode & ~hob.session_initialized)
def collect_act_event(val = True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.profile_conf.collect_act_event = val
