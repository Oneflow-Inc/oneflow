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
import os
import traceback

import oneflow._oneflow_internal
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.framework.hob as hob
import oneflow.framework.session_context as session_ctx
import oneflow.support.enable_if as enable_if


def _set_attr_to_resource(attr_name, attr_value):
    sess = session_ctx.GetDefaultSession()
    if sess.status_ == sess.Status.INITED:
        reso_config = resource_util.Resource()
        setattr(reso_config, attr_name, attr_value)
        sess.update_resource_eagerly(reso_config)
    else:
        setattr(sess.config_proto.resource, attr_name, attr_value)


def api_load_library(val: str) -> None:
    """Load necessary library for job

    Args:
        val (str): path to shared object file
    """
    return enable_if.unique([load_library, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def load_library(val):
    assert type(val) is str
    sess = session_ctx.GetDefaultSession()
    sess.config_proto.load_lib_path.append(val)


def api_load_library_now(val: str) -> None:
    """Load necessary library for job now

    Args:
        val (str): path to shared object file
    """
    return enable_if.unique([load_library_now, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def load_library_now(val):
    assert type(val) is str
    oneflow._oneflow_internal.LoadLibraryNow(val)


def api_machine_num(val: int) -> None:
    """Set available number of machine/node for  running job .

    Args:
        val (int): available number of machines
    """
    return enable_if.unique([machine_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def machine_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.machine_num = val


def api_gpu_device_num(val: int) -> None:
    """Set number of GPUs on each machine to run oneflow on.

    Args:
        val (int): number of GPUs. It is identical on every machine. In other words,
        you can't specify different number of GPUs you would like to use on each machine.
    """
    if oneflow._oneflow_internal.flags.with_cuda():
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


def api_cpu_device_num(val: int) -> None:
    """Set number of CPUs on each machine to run oneflow on. Usually you don't need to set this.

    Args:
        val (int): number of CPUs. It is identical on every machine.
    """
    return enable_if.unique([cpu_device_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def cpu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.cpu_device_num = val


def api_comm_net_worker_num(val: int) -> None:
    """Set up the workers number in epoll  mode network,
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


def api_max_mdsave_worker_num(val: int) -> None:
    """Set up max number of workers for mdsave process.

    Args:
        val (int):  max number of workers
    """
    return enable_if.unique([max_mdsave_worker_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def max_mdsave_worker_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.max_mdsave_worker_num = val


def api_numa_aware_cuda_malloc_host(val: bool = True) -> None:
    """Whether or not let numa know  that  cuda allocated host's memory.

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    print(
        "'enable_numa_aware_cuda_malloc_host' has been deprecated, has no effect and will be removed in the future."
    )


def api_compute_thread_pool_size(val: int) -> None:
    """Set up the size of compute thread pool

    Args:
        val (int): size of  thread pool
    """
    return enable_if.unique([compute_thread_pool_size, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def compute_thread_pool_size(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.compute_thread_pool_size = val


def api_reserved_host_mem_mbyte(val: int) -> None:
    """Set up the memory size of reserved host

    Args:
        val (int):  memory size, e.g. 1024(mb)
    """
    return enable_if.unique([reserved_host_mem_mbyte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def reserved_host_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_host_mem_mbyte = val


def api_reserved_device_mem_mbyte(val: int) -> None:
    """Set up the memory size of reserved device

    Args:
        val (int):  memory size, e.g. 1024(mb)
    """
    return enable_if.unique([reserved_device_mem_mbyte, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def reserved_device_mem_mbyte(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.reserved_device_mem_mbyte = val


def api_enable_cudnn_fused_normalization_add_relu(val: bool) -> None:
    """Whether enable cudnn_fused_normalization_add_relu.

    Args:
        val (bool): whether enable or not
    """
    return enable_if.unique([enable_cudnn_fused_normalization_add_relu, do_nothing])(
        val
    )


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_cudnn_fused_normalization_add_relu(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.cudnn_conf.enable_cudnn_fused_normalization_add_relu = (
        val
    )


def api_enable_debug_mode(val: bool) -> None:
    """Whether use debug mode or not.

    Args:
        val (bool):  True or False
    """
    return enable_if.unique([enable_debug_mode, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_debug_mode(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_debug_mode = val


def api_legacy_model_io_enabled():
    sess = session_ctx.GetDefaultSession()
    return sess.config_proto.resource.enable_legacy_model_io


def api_enable_legacy_model_io(val: bool = True):
    """Whether or not use legacy model io.

    Args:
        val ([type]): True or False
    """
    return enable_if.unique([enable_legacy_model_io, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_legacy_model_io(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_legacy_model_io = val


def api_enable_model_io_v2(val):
    """Whether or not use version2  of model input/output function.

    Args:
        val ([type]): True or False
    """
    return enable_if.unique([enable_model_io_v2, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_model_io_v2(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_model_io_v2 = val


def api_enable_fusion(val: bool = True) -> None:
    """Whether or not allow fusion the operators

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([enable_fusion, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_fusion(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.enable_fusion = val


def api_num_callback_threads(val: int) -> None:
    """Set up number of callback threads for boxing process.
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


def api_enable_tensor_float_32_compute(val: bool = True) -> None:
    """Whether or not to enable Tensor-float-32 on supported GPUs

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([enable_tensor_float_32_compute, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_tensor_float_32_compute(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_tensor_float_32_compute = val
    if not val:
        os.environ["ONEFLOW_EP_CUDA_ENABLE_TF32_EXECUTION"] = "0"


def api_enable_mem_chain_merge(val: bool = True) -> None:
    """Whether or not to enable MemChain merge.

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    return enable_if.unique([enable_mem_chain_merge, do_nothing])(val=val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def enable_mem_chain_merge(val=True):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.enable_mem_chain_merge = val


def api_nccl_use_compute_stream(val: bool = False) -> None:
    """Whether or not nccl use compute stream to reuse nccl memory and speedup

    Args:
        val (bool, optional): True or False. Defaults to False.
    """
    assert type(val) is bool
    _set_attr_to_resource("nccl_use_compute_stream", val)


def api_disable_group_boxing_by_dst_parallel(val: bool = False) -> None:
    """Whether or not disable group boxing by dst parallel pass to reduce boxing memory life cycle.

    Args:
        val (bool, optional): True or False. Defaults to False.
    """
    assert type(val) is bool
    _set_attr_to_resource("disable_group_boxing_by_dst_parallel", val)


def api_nccl_num_streams(val: int) -> None:
    """Set up the number of nccl parallel streams while use boxing

    Args:
        val (int): number of streams
    """
    return enable_if.unique([nccl_num_streams, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_num_streams(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_num_streams = val


def api_nccl_fusion_threshold_mb(val: int) -> None:
    """Set up threshold for oprators fusion

    Args:
        val (int): int number, e.g. 10(mb)
    """
    return enable_if.unique([nccl_fusion_threshold_mb, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_threshold_mb(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_threshold_mb = val


def api_nccl_fusion_all_reduce_use_buffer(val: bool) -> None:
    """Whether or not use buffer during nccl fusion progress

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


def api_nccl_fusion_all_reduce(val: bool) -> None:
    """Whether or not use nccl fusion during all reduce progress

    Args:
        val (bool):  True or False
    """
    return enable_if.unique([nccl_fusion_all_reduce, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_reduce(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_reduce = val


def api_nccl_fusion_reduce_scatter(val: bool) -> None:
    """Whether or not  use nccl fusion during reduce scatter progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_reduce_scatter, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_reduce_scatter(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_reduce_scatter = val


def api_nccl_fusion_all_gather(val: bool) -> None:
    """Whether or not use nccl fusion during all  gather progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_all_gather, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_all_gather(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_all_gather = val


def api_nccl_fusion_reduce(val: bool) -> None:
    """Whether or not use nccl fusion during reduce progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_reduce, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_reduce(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_reduce = val


def api_nccl_fusion_broadcast(val: bool) -> None:
    """Whether or not use nccl fusion during broadcast progress

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_fusion_broadcast, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_broadcast(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_broadcast = val


def api_nccl_fusion_max_ops(val: int) -> None:
    """Maximum number of ops for nccl fusion.

    Args:
        val (int): Maximum number of ops
    """
    return enable_if.unique([nccl_fusion_max_ops, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_fusion_max_ops(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.collective_boxing_conf.nccl_fusion_max_ops = val


def api_nccl_enable_all_to_all(val: bool) -> None:
    """Whether or not use nccl all2all during s2s boxing

    Args:
        val (bool): True or False
    """
    return enable_if.unique([nccl_enable_all_to_all, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def nccl_enable_all_to_all(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is bool
    sess.config_proto.resource.collective_boxing_conf.nccl_enable_all_to_all = val


def api_nccl_enable_mixed_fusion(val: bool) -> None:
    """Whether or not use nccl mixed fusion

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
    raise NotImplementedError(
        "This action donot working because session is initialized."
    )
