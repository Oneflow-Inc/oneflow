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
import sys
import traceback
from typing import List, Union

import oneflow._oneflow_internal
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.framework.hob as hob
import oneflow.framework.session_context as session_ctx
import oneflow.support.enable_if as enable_if


def _set_resource_attr(attrs_chain: Union[List[str], str], attr_value):

    if isinstance(attrs_chain, str):
        attrs_chain = [attrs_chain]

    session = session_ctx.GetDefaultSession()

    def __get_obj(obj, _attrs_chain):
        last_obj = obj
        for att in _attrs_chain:
            last_obj = getattr(last_obj, att)
        return last_obj

    if session.status_ == session.Status.INITED:
        resource_config = resource_util.Resource()
        setattr(
            __get_obj(resource_config, attrs_chain[0:-1]), attrs_chain[-1], attr_value
        )
        session.update_resource_eagerly(resource_config)
    else:
        setattr(
            __get_obj(session.config_proto.resource, attrs_chain[0:-1]),
            attrs_chain[-1],
            attr_value,
        )


def api_machine_num(val: int) -> None:
    """Set available number of machine/node for  running job .

    Args:
        val (int): available number of machines
    """
    assert type(val) is int
    _set_resource_attr("machine_num", val)


def api_gpu_device_num(val: int) -> None:
    """Set number of GPUs on each machine to run oneflow on.

    Args:
        val (int): number of GPUs. It is identical on every machine. In other words,
        you can't specify different number of GPUs you would like to use on each machine.
    """
    assert type(val) is int
    if oneflow._oneflow_internal.flags.with_cuda():
        _set_resource_attr("gpu_device_num", val)
    else:
        print(
            "INFO: for CPU-only OneFlow, oneflow.config.gpu_device_num is equivalent to oneflow.config.cpu_device_num"
        )
        print(traceback.format_stack()[-2])
        _set_resource_attr("cpu_device_num", val)


def api_cpu_device_num(val: int) -> None:
    """Set number of CPUs on each machine to run oneflow on. Usually you don't need to set this.

    Args:
        val (int): number of CPUs. It is identical on every machine.
    """
    assert type(val) is int
    _set_resource_attr("cpu_device_num", val)


def api_comm_net_worker_num(val: int) -> None:
    """Set up the workers number in epoll  mode network,
            If use RDMA mode network, then doesn't need.

    Args:
        val (int): number of workers
    """
    assert type(val) is int
    _set_resource_attr("comm_net_worker_num", val)


def api_max_mdsave_worker_num(val: int) -> None:
    """Set up max number of workers for mdsave process.

    Args:
        val (int):  max number of workers
    """
    assert type(val) is int
    _set_resource_attr("max_mdsave_worker_num", val)


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
    assert type(val) is int
    _set_resource_attr("compute_thread_pool_size", val)


def api_reserved_host_mem_mbyte(val: int) -> None:
    """Set up the memory size of reserved host

    Args:
        val (int):  memory size, e.g. 1024(mb)
    """
    assert type(val) is int
    _set_resource_attr("reserved_host_mem_mbyte", val)


def api_reserved_device_mem_mbyte(val: int) -> None:
    """Set up the memory size of reserved device

    Args:
        val (int):  memory size, e.g. 1024(mb)
    """
    assert type(val) is int
    _set_resource_attr("reserved_device_mem_mbyte", val)


def api_enable_cudnn_fused_normalization_add_relu(val: bool) -> None:
    """Whether enable cudnn_fused_normalization_add_relu.

    Args:
        val (bool): whether enable or not
    """
    assert type(val) is bool
    _set_resource_attr(["cudnn_conf", "enable_cudnn_fused_normalization_add_relu"], val)


def api_enable_debug_mode(val: bool) -> None:
    """Whether use debug mode or not.

    Args:
        val (bool):  True or False
    """
    assert type(val) is bool
    _set_resource_attr("enable_debug_mode", val)


def api_legacy_model_io_enabled():
    sess = session_ctx.GetDefaultSession()
    return sess.config_proto.resource.enable_legacy_model_io


def api_enable_legacy_model_io(val: bool = True):
    """Whether or not use legacy model io.

    Args:
        val ([type]): True or False
    """
    assert type(val) is bool
    _set_resource_attr("enable_legacy_model_io", val)


def api_enable_model_io_v2(val: bool):
    """Whether or not use version2  of model input/output function.

    Args:
        val ([type]): True or False
    """
    assert type(val) is bool
    _set_resource_attr("enable_legacy_model_io", val)


def api_enable_fusion(val: bool = True) -> None:
    """Whether or not allow fusion the operators

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "enable_fusion"], val)


def api_num_callback_threads(val: int) -> None:
    """Set up number of callback threads for boxing process.
            Boxing is used to convert between different parallel properties of logical tensor

    Args:
        val (int): number of  callback threads
    """
    assert type(val) is int
    _set_resource_attr(["collective_boxing_conf", "num_callback_threads"], val)


def api_enable_tensor_float_32_compute(val: bool = True) -> None:
    """Whether or not to enable Tensor-float-32 on supported GPUs

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    assert type(val) is bool
    _set_resource_attr("enable_tensor_float_32_compute", val)
    if not val:
        os.environ["ONEFLOW_EP_CUDA_ENABLE_TF32_EXECUTION"] = "0"


def api_enable_mem_chain_merge(val: bool = True) -> None:
    """Whether or not to enable MemChain merge.

    Args:
        val (bool, optional): True or False. Defaults to True.
    """
    assert type(val) is bool
    _set_resource_attr("enable_mem_chain_merge", val)


def api_nccl_use_compute_stream(val: bool = False) -> None:
    """Whether or not nccl use compute stream to reuse nccl memory and speedup

    Args:
        val (bool, optional): True or False. Defaults to False.
    """
    assert type(val) is bool
    _set_resource_attr("nccl_use_compute_stream", val)


def api_disable_group_boxing_by_dst_parallel(val: bool = False) -> None:
    """Whether or not disable group boxing by dst parallel pass to reduce boxing memory life cycle.

    Args:
        val (bool, optional): True or False. Defaults to False.
    """
    assert type(val) is bool
    _set_resource_attr("disable_group_boxing_by_dst_parallel", val)


def api_nccl_num_streams(val: int) -> None:
    """Set up the number of nccl parallel streams while use boxing

    Args:
        val (int): number of streams
    """
    assert type(val) is int
    _set_resource_attr(["collective_boxing_conf", "nccl_num_streams"], val)


def api_nccl_fusion_threshold_mb(val: int) -> None:
    """Set up threshold for oprators fusion

    Args:
        val (int): int number, e.g. 10(mb)
    """
    assert type(val) is int
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_threshold_mb"], val)


def api_nccl_fusion_all_reduce_use_buffer(val: bool) -> None:
    """Whether or not use buffer during nccl fusion progress

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(
        ["collective_boxing_conf", "nccl_fusion_all_reduce_use_buffer"], val
    )


def api_nccl_fusion_all_reduce(val: bool) -> None:
    """Whether or not use nccl fusion during all reduce progress

    Args:
        val (bool):  True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_all_reduce"], val)


def api_nccl_fusion_reduce_scatter(val: bool) -> None:
    """Whether or not  use nccl fusion during reduce scatter progress

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_reduce_scatter"], val)


def api_nccl_fusion_all_gather(val: bool) -> None:
    """Whether or not use nccl fusion during all  gather progress

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_all_gather"], val)


def api_nccl_fusion_reduce(val: bool) -> None:
    """Whether or not use nccl fusion during reduce progress

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_reduce"], val)


def api_nccl_fusion_broadcast(val: bool) -> None:
    """Whether or not use nccl fusion during broadcast progress

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_broadcast"], val)


def api_nccl_fusion_max_ops(val: int) -> None:
    """Maximum number of ops for nccl fusion.

    Args:
        val (int): Maximum number of ops
    """
    assert type(val) is int
    _set_resource_attr(["collective_boxing_conf", "nccl_fusion_max_ops"], val)


def api_nccl_enable_all_to_all(val: bool) -> None:
    """Whether or not use nccl all2all during s2s boxing

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_enable_all_to_all"], val)


def api_nccl_enable_mixed_fusion(val: bool) -> None:
    """Whether or not use nccl mixed fusion

    Args:
        val (bool): True or False
    """
    assert type(val) is bool
    _set_resource_attr(["collective_boxing_conf", "nccl_enable_mixed_fusion"], val)


@enable_if.condition(hob.in_normal_mode & hob.session_initialized)
def do_nothing(*args, **kwargs):
    print("This action donot working because session is initialized.", file=sys.stderr)
