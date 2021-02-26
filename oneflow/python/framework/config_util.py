"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import, print_function

import oneflow.python.framework.hob as hob
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api
import traceback


@oneflow_export("config.load_library")
def api_load_library(val: str) -> None:
    r"""Load necessary library for job

    Args:
        val (str): path to shared object file
    """
    return enable_if.unique([load_library, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def load_library(val):
    assert type(val) is str
    sess = session_ctx.GetDefaultSession()
    sess.config_proto.load_lib_path.append(val)


@oneflow_export("config.load_library_now")
def api_load_library_now(val: str) -> None:
    r"""Load necessary library for job now

    Args:
        val (str): path to shared object file
    """
    return enable_if.unique([load_library_now, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def load_library_now(val):
    assert type(val) is str
    oneflow_api.LoadLibraryNow(val)


@oneflow_export("config.machine_num")
def api_machine_num(val: int) -> None:
    r"""Set available number of machine/node for  running job .

    Args:
        val (int): available number of machines
    """
    return enable_if.unique([machine_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def machine_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.machine_num = val


@oneflow_export("config.gpu_device_num")
def api_gpu_device_num(val: int) -> None:
    r"""Set number of GPUs on each machine to run oneflow on.

    Args:
        val (int): number of GPUs. It is identical on every machine. In other words,
        you can't specify different number of GPUs you would like to use on each machine.
    """
    if oneflow_api.flags.with_cuda():
        return enable_if.unique([gpu_device_num, do_nothing])(val)
    else:
        print(
            "INFO: for CPU-only OneFlow, oneflow.config.gpu_device_num is equivalent to oneflow.config.cpu_device_num"
        )
        print(traceback.format_stack()[-2])
        return enable_if.unique([cpu_device_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def gpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.gpu_device_num = val


@oneflow_export("config.cpu_device_num")
def api_cpu_device_num(val: int) -> None:
    r"""Set number of CPUs on each machine to run oneflow on. Usually you don't need to set this.

    Args:
        val (int): number of CPUs. It is identical on every machine.
    """
    return enable_if.unique([cpu_device_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def cpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.cpu_device_num = val


@oneflow_export("config.comm_net_worker_num")
def api_comm_net_worker_num(val: int) -> None:
    r"""Set up the workers number in epoll  mode network,
            If use RDMA mode network, then doesn't need.

    Args:
        val (int): number of workers
    """
    return enable_if.unique([comm_net_worker_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def comm_net_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.comm_net_worker_num = val


@oneflow_export("config.max_mdsave_worker_num")
def api_max_mdsave_worker_num(val: int) -> None:
    r"""Set up max number of workers for mdsave process.

    Args:
        val (int):  max number of workers
    """
    return enable_if.unique([max_mdsave_worker_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def max_mdsave_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.max_mdsave_worker_num = val


@oneflow_export("config.enable_numa_aware_cuda_malloc_host")
def api_numa_aware_cuda_malloc_host(val: bool = True) -> None:
    r"""Whether or not let numa know  that  cuda allocated host's memory.

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([enable_numa_aware_cuda_malloc_host, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_numa_aware_cuda_malloc_host(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_numa_aware_cuda_malloc_host = val


@oneflow_export("config.compute_thread_pool_size")
def api_compute_thread_pool_size(val: int) -> None:
    r"""Set up the size of compute thread pool

    Args:
        val (int): size of  thread pool
    """
    return enable_if.unique([compute_thread_pool_size, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def compute_thread_pool_size(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.compute_thread_pool_size = val


@oneflow_export("config.rdma_mem_block_mbyte")
def api_rdma_mem_block_mbyte(val: int) -> None:
    r"""Set up the memory block size in rdma mode.

    Args:
        val (int): size of block, e.g. 1024(mb)
    """
    return enable_if.unique([rdma_mem_block_mbyte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def rdma_mem_block_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.rdma_mem_block_mbyte = val


@oneflow_export("config.rdma_recv_msg_buf_mbyte")
def api_rdma_recv_msg_buf_mbyte(val: int) -> None:
    r"""Set up the buffer size for receiving messages in rama mode

    Args:
        val (int): buffer size, e.g. 1024(mb)
    """
    return enable_if.unique([rdma_recv_msg_buf_mbyte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def rdma_recv_msg_buf_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.rdma_recv_msg_buf_mbyte = val


@oneflow_export("config.reserved_host_mem_mbyte")
def api_reserved_host_mem_mbyte(val: int) -> None:
    r"""Set up the memory size of reserved host

    Args:
        val (int):  memory size, e.g. 1024(mb)
    """
    return enable_if.unique([reserved_host_mem_mbyte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def reserved_host_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_host_mem_mbyte = val


@oneflow_export("config.reserved_device_mem_mbyte")
def api_reserved_device_mem_mbyte(val: int) -> None:
    r"""Set up the memory size of reserved device

    Args:
        val (int):  memory size, e.g. 1024(mb)
    """
    return enable_if.unique([reserved_device_mem_mbyte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def reserved_device_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_device_mem_mbyte = val


@oneflow_export("config.use_rdma")
def api_use_rdma(val: bool = True) -> None:
    r"""Whether use RDMA to speed up data transmission in cluster nodes or not.
          if not, then use normal epoll mode.

    Args:
        val (bool, optional):  Defaults to True.
    """
    return enable_if.unique([use_rdma, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def use_rdma(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.use_rdma = val


@oneflow_export("config.thread_enable_local_message_queue")
def api_thread_enable_local_message_queue(val: bool) -> None:
    """Whether or not enable thread using local  message queue.

    Args:
        val (bool):  True or False
    """
    return enable_if.unique([thread_enable_local_message_queue, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def thread_enable_local_message_queue(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.thread_enable_local_message_queue = val


@oneflow_export("config.enable_debug_mode")
def api_enable_debug_mode(val: bool) -> None:
    r"""Whether use debug mode or not.

    Args:
        val (bool):  True or False
    """
    return enable_if.unique([enable_debug_mode, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_debug_mode(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_debug_mode = val


@oneflow_export("config.save_downloaded_file_to_local_fs")
def api_save_downloaded_file_to_local_fs(val: bool = True) -> None:
    r"""Whether or not save downloaded file to local file system.

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([save_downloaded_file_to_local_fs, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def save_downloaded_file_to_local_fs(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.io_conf.save_downloaded_file_to_local_fs = val


@oneflow_export("config.persistence_buf_byte")
def api_persistence_buf_byte(val: int) -> None:
    r"""Set up buffer size for persistence.

    Args:
        val (int): e.g. 1024(bytes)
    """
    return enable_if.unique([persistence_buf_byte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def persistence_buf_byte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.io_conf.persistence_buf_byte = val


@oneflow_export("config.legacy_model_io_enabled")
def api_legacy_model_io_enabled():
    sess = session_ctx.GetDefaultSession()
    return sess.config_proto.io_conf.enable_legacy_model_io


@oneflow_export("config.enable_legacy_model_io")
def api_enable_legacy_model_io(val: bool = True):
    r"""Whether or not use legacy model io.

    Args:
        val ([type]): True or False
    """
    return enable_if.unique([enable_legacy_model_io, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_legacy_model_io(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.io_conf.enable_legacy_model_io = val


@oneflow_export("config.enable_model_io_v2")
def api_enable_model_io_v2(val):
    r"""Whether or not use version2  of model input/output function.

    Args:
        val ([type]): True or False
    """
    return enable_if.unique([enable_model_io_v2, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_model_io_v2(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.io_conf.enable_model_io_v2 = val


@oneflow_export("config.collect_act_event")
def api_collect_act_event(val: bool = True) -> None:
    r"""Whether or not collect active event.

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([collect_act_event, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def collect_act_event(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.profile_conf.collect_act_event = val


@oneflow_export("config.collective_boxing.enable_fusion")
def api_enable_fusion(val: bool = True) -> None:
    r"""Whether or not allow fusion the operators

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([enable_fusion, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_fusion(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.enable_fusion = val


@oneflow_export("config.collective_boxing.num_callback_threads")
def api_num_callback_threads(val: int) -> None:
    r"""Set up number of callback threads for boxing process.
            Boxing is used to convert between different parallel properties of logical tensor

    Args:
        val (int): number of  callback threads
    """
    return enable_if.unique([num_callback_threads, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def num_callback_threads(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.num_callback_threads = val


@oneflow_export("config.enable_tensor_float_32_compute")
def api_enable_tensor_float_32_compute(val: bool = True) -> None:
    r"""Whether or not to enable Tensor-float-32 on supported GPUs

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([enable_tensor_float_32_compute, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_tensor_float_32_compute(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_tensor_float_32_compute = val


@oneflow_export("config.nccl_use_compute_stream")
def api_nccl_use_compute_stream(val: bool = False) -> None:
    r"""Whether or not nccl use compute stream to reuse nccl memory and speedup

    Args:
        val (bool, optional): True or False. Defaults to False.
    """
    return enable_if.unique([nccl_use_compute_stream, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_use_compute_stream(val=False):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.nccl_use_compute_stream = val


@oneflow_export("config.disable_group_boxing_by_dst_parallel")
def api_disable_group_boxing_by_dst_parallel(val: bool = False) -> None:
    r"""Whether or not disable group boxing by dst parallel pass to reduce boxing memory life cycle.

    Args:
        val (bool, optional): True or False. Defaults to False.
    """
    return enable_if.unique([disable_group_boxing_by_dst_parallel, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def disable_group_boxing_by_dst_parallel(val=False):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.disable_group_boxing_by_dst_parallel = val


@oneflow_export("config.collective_boxing.nccl_num_streams")
def api_nccl_num_streams(val: int) -> None:
    r"""Set up the number of nccl parallel streams while use boxing

    Args:
        val (int): number of streams
    """
    return enable_if.unique([nccl_num_streams, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_num_streams(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_num_streams = val


@oneflow_export("config.collective_boxing.nccl_fusion_threshold_mb")
def api_nccl_fusion_threshold_mb(val: int) -> None:
    r"""Set up threshold for oprators fusion

    Args:
        val (int): int number, e.g. 10(mb)
    """
    return enable_if.unique([nccl_fusion_threshold_mb, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_threshold_mb(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_threshold_mb = val


@oneflow_export("config.collective_boxing.nccl_fusion_all_reduce_use_buffer")
def api_nccl_fusion_all_reduce_use_buffer(val: bool) -> None:
    r"""Whether or not use buffer during nccl fusion progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_all_reduce_use_buffer, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_reduce_use_buffer(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_reduce_use_buffer = (
        val
    )


@oneflow_export("config.collective_boxing.nccl_fusion_all_reduce")
def api_nccl_fusion_all_reduce(val: bool) -> None:
    r"""Whether or not use nccl fusion during all reduce progress

    Args:
        val (bool):  True or False
    """
    return enable_if.unique([nccl_fusion_all_reduce, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_reduce(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_reduce = val


@oneflow_export("config.collective_boxing.nccl_fusion_reduce_scatter")
def api_nccl_fusion_reduce_scatter(val: bool) -> None:
    r"""Whether or not  use nccl fusion during reduce scatter progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_reduce_scatter, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_reduce_scatter(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_reduce_scatter = val


@oneflow_export("config.collective_boxing.nccl_fusion_all_gather")
def api_nccl_fusion_all_gather(val: bool) -> None:
    r"""Whether or not use nccl fusion during all  gather progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_all_gather, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_gather(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_gather = val


@oneflow_export("config.collective_boxing.nccl_fusion_reduce")
def api_nccl_fusion_reduce(val: bool) -> None:
    r"""Whether or not use nccl fusion during reduce progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_reduce, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_reduce(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_reduce = val


@oneflow_export("config.collective_boxing.nccl_fusion_broadcast")
def api_nccl_fusion_broadcast(val: bool) -> None:
    r"""Whether or not use nccl fusion during broadcast progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_broadcast, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_broadcast(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_broadcast = val


@oneflow_export("config.collective_boxing.nccl_fusion_max_ops")
def api_nccl_fusion_max_ops(val: int) -> None:
    r"""Maximum number of ops for nccl fusion.

    Args:
        val (int): Maximum number of ops
    """
    return enable_if.unique([nccl_fusion_max_ops, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_max_ops(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_max_ops = val


@oneflow_export("config.collective_boxing.nccl_enable_all_to_all")
def api_nccl_enable_all_to_all(val: bool) -> None:
    r"""Whether or not use nccl all2all during s2s boxing

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_enable_all_to_all, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_enable_all_to_all(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_enable_all_to_all = val


@oneflow_export("config.collective_boxing.nccl_enable_mixed_fusion")
def api_nccl_enable_mixed_fusion(val: bool) -> None:
    r"""Whether or not use nccl mixed fusion

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_enable_mixed_fusion, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_enable_mixed_fusion(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_enable_mixed_fusion = val


@enable_if.condition(hob.in_normal_mode & hob.session_initialized)
def do_nothing(*args, **kwargs):
    print("Nothing happened because the session is running")
    return False
