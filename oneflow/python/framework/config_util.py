from __future__ import absolute_import, print_function

import oneflow.python.framework.hob as hob
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("config.load_library")
def api_load_library(val):
    return enable_if.unique(load_library, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def load_library(val):
    assert type(val) is str
    sess = session_ctx.GetDefaultSession()
    sess.config_proto.load_lib_path.append(val)


@oneflow_export("config.machine_num")
def api_machine_num(val):
    return enable_if.unique(machine_num, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def machine_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.machine_num = val


@oneflow_export("config.gpu_device_num")
def api_gpu_device_num(val):
    r"""Set number of GPUs on each machine to run oneflow on.

    Args:
        val (int): number of GPUs. It is identical on every machine. In other words, you can't specify different number of GPUs you would like to use on each machine.
    """
    return enable_if.unique(gpu_device_num, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def gpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.gpu_device_num = val


@oneflow_export("config.cpu_device_num")
def api_cpu_device_num(val):
    r"""Set number of CPUs on each machine to run oneflow on. Usually you don't need to set this.

    Args:
        val (int): number of CPUs. It is identical on every machine.
    """
    return enable_if.unique(cpu_device_num, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def cpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.cpu_device_num = val


@oneflow_export("config.comm_net_worker_num")
def api_comm_net_worker_num(val):
    return enable_if.unique(comm_net_worker_num, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def comm_net_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.comm_net_worker_num = val


@oneflow_export("config.max_mdsave_worker_num")
def api_max_mdsave_worker_num(val):
    return enable_if.unique(max_mdsave_worker_num, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def max_mdsave_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.max_mdsave_worker_num = val


@oneflow_export("config.enable_numa_aware_cuda_malloc_host")
def api_numa_aware_cuda_malloc_host(val=True):
    return enable_if.unique(enable_numa_aware_cuda_malloc_host, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_numa_aware_cuda_malloc_host(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_numa_aware_cuda_malloc_host = val


@oneflow_export("config.compute_thread_pool_size")
def api_compute_thread_pool_size(val):
    return enable_if.unique(compute_thread_pool_size, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def compute_thread_pool_size(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.compute_thread_pool_size = val


@oneflow_export("config.rdma_mem_block_mbyte")
def api_rdma_mem_block_mbyte(val):
    return enable_if.unique(rdma_mem_block_mbyte, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def rdma_mem_block_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.rdma_mem_block_mbyte = val


@oneflow_export("config.rdma_recv_msg_buf_mbyte")
def api_rdma_recv_msg_buf_mbyte(val):
    return enable_if.unique(rdma_recv_msg_buf_mbyte, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def rdma_recv_msg_buf_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.rdma_recv_msg_buf_mbyte = val


@oneflow_export("config.reserved_host_mem_mbyte")
def api_reserved_host_mem_mbyte(val):
    return enable_if.unique(reserved_host_mem_mbyte, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def reserved_host_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_host_mem_mbyte = val


@oneflow_export("config.reserved_device_mem_mbyte")
def api_reserved_device_mem_mbyte(val):
    return enable_if.unique(reserved_device_mem_mbyte, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def reserved_device_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_device_mem_mbyte = val


@oneflow_export("config.use_rdma")
def api_use_rdma(val=True):
    return enable_if.unique(use_rdma, do_nothing)(val=True)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def use_rdma(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.use_rdma = val


@oneflow_export("config.thread_enable_local_message_queue")
def api_thread_enable_local_message_queue(val):
    return enable_if.unique(thread_enable_local_message_queue, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def thread_enable_local_message_queue(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.thread_enable_local_message_queue = val


@oneflow_export("config.enable_debug_mode")
def api_enable_debug_mode(val):
    return enable_if.unique(enable_debug_mode, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_debug_mode(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_debug_mode = val


@oneflow_export("config.save_downloaded_file_to_local_fs")
def api_save_downloaded_file_to_local_fs(val=True):
    return enable_if.unique(save_downloaded_file_to_local_fs, do_nothing)(val=True)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def save_downloaded_file_to_local_fs(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.io_conf.save_downloaded_file_to_local_fs = val


@oneflow_export("config.persistence_buf_byte")
def api_persistence_buf_byte(val):
    return enable_if.unique(persistence_buf_byte, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def persistence_buf_byte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.io_conf.persistence_buf_byte = val


@oneflow_export("config.collect_act_event")
def api_collect_act_event(val=True):
    return enable_if.unique(collect_act_event, do_nothing)(val=True)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def collect_act_event(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.profile_conf.collect_act_event = val


@oneflow_export("config.collective_boxing.enable_fusion")
def api_enable_fusion(val=True):
    return enable_if.unique(enable_fusion, do_nothing)(val=True)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_fusion(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.enable_fusion = val


@oneflow_export("config.collective_boxing.num_callback_threads")
def api_num_callback_threads(val):
    return enable_if.unique(num_callback_threads, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def num_callback_threads(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.num_callback_threads = val


@oneflow_export("config.collective_boxing.nccl_num_streams")
def api_nccl_num_streams(val):
    return enable_if.unique(nccl_num_streams, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_num_streams(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_num_streams = val


@oneflow_export("config.collective_boxing.nccl_fusion_threshold_mb")
def api_nccl_fusion_threshold_mb(val):
    return enable_if.unique(nccl_fusion_threshold_mb, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_threshold_mb(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_threshold_mb = val


@oneflow_export("config.collective_boxing.nccl_fusion_all_reduce_use_buffer")
def api_nccl_fusion_all_reduce_use_buffer(val):
    return enable_if.unique(nccl_fusion_all_reduce_use_buffer, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_reduce_use_buffer(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_reduce_use_buffer = (
        val
    )


@oneflow_export("config.collective_boxing.nccl_fusion_all_reduce")
def api_nccl_fusion_all_reduce(val):
    return enable_if.unique(nccl_fusion_all_reduce, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_reduce(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_reduce = val


@oneflow_export("config.collective_boxing.nccl_fusion_reduce_scatter")
def api_nccl_fusion_reduce_scatter(val):
    return enable_if.unique(nccl_fusion_reduce_scatter, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_reduce_scatter(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_reduce_scatter = val


@oneflow_export("config.collective_boxing.nccl_fusion_all_gather")
def api_nccl_fusion_all_gather(val):
    return enable_if.unique(nccl_fusion_all_gather, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_gather(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_gather = val


@oneflow_export("config.collective_boxing.nccl_fusion_reduce")
def api_nccl_fusion_reduce(val):
    return enable_if.unique(nccl_fusion_reduce, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_reduce(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_reduce = val


@oneflow_export("config.collective_boxing.nccl_fusion_broadcast")
def api_nccl_fusion_broadcast(val):
    return enable_if.unique(nccl_fusion_broadcast, do_nothing)(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_broadcast(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_broadcast = val


@enable_if.condition(hob.in_normal_mode & hob.session_initialized)
def do_nothing(*args, **kwargs):
    print("Nothing happened because the session is running")
    return False
