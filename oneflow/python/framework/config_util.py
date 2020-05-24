from __future__ import absolute_import
from __future__ import print_function

import sys
import oneflow
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.job.job_set_pb2 as job_set_pb
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.c_api_util as c_api_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.pb_util as pb_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if

@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def load_libray(val):
    assert type(val) is str
    sess = session_ctx.GetDefaultSession()
    sess.config_proto.load_lib_path.append(val)

@oneflow_export('config.load_library')
def api_load_libray(val):
    return enable_if.unique(load_libray)(val)

@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def machine_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.machine_num = val

@oneflow_export('config.machine_num')
def api_machine_num(val):
    return enable_if.unique(machine_num)(val)

@oneflow_export('config.gpu_device_num')
def gpu_device_num(val):
    r"""Set number of GPUs on each machine to run oneflow on.

    Args:
        val: number of GPUs. It is identical on every machine. In other words, you can't specify different number of GPUs you would like to use on each machine.
    """
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.gpu_device_num = val

@oneflow_export('config.cpu_device_num')
def cpu_device_num(val):
    r"""Set number of CPUs on each machine to run oneflow on. Usually you don't need to set this.

    Args:
        val: number of CPUs. It is identical on every machine.
    """
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.cpu_device_num = val

@oneflow_export('config.comm_net_worker_num')
def comm_net_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.comm_net_worker_num = val

@oneflow_export('config.max_mdsave_worker_num')
def max_mdsave_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.max_mdsave_worker_num = val

@oneflow_export('config.compute_thread_pool_size')
def compute_thread_pool_size(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.compute_thread_pool_size = val

@oneflow_export('config.rdma_mem_block_mbyte')
def rdma_mem_block_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.rdma_mem_block_mbyte = val

@oneflow_export('config.rdma_recv_msg_buf_mbyte')
def rdma_recv_msg_buf_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.rdma_recv_msg_buf_mbyte = val

@oneflow_export('config.reserved_host_mem_mbyte')
def reserved_host_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.reserved_host_mem_mbyte = val

@oneflow_export('config.reserved_device_mem_mbyte')
def reserved_device_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.resource.reserved_device_mem_mbyte = val

@oneflow_export('config.use_rdma')
def use_rdma(val = True):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is bool
    sess.config_proto.resource.use_rdma = val

@oneflow_export('config.thread_enable_local_message_queue')
def thread_enable_local_message_queue(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is bool
    sess.config_proto.resource.thread_enable_local_message_queue = val


@oneflow_export('config.enable_debug_mode')
def enable_debug_mode(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is bool
    sess.config_proto.resource.enable_debug_mode = val

@oneflow_export('config.save_downloaded_file_to_local_fs')
def save_downloaded_file_to_local_fs(val = True):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is bool
    sess.config_proto.io_conf.save_downloaded_file_to_local_fs = val

@oneflow_export('config.persistence_buf_byte')
def persistence_buf_byte(val):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.io_conf.persistence_buf_byte = val

@oneflow_export('config.collect_act_event')
def collect_act_event(val = True):
    sess = session_ctx.GetDefaultSession()
    if sess.is_running:
        print("flow.config.* are disabled when session running", file=sys.stderr)
        return
    assert type(val) is int
    sess.config_proto.profile_conf.collect_act_event = val
