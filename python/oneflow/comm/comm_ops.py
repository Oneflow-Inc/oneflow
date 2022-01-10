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

import oneflow as flow
import numpy as np


def all_reduce(tensor):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.
    After the call ``tensor`` is going to be bitwise identical in all processes.

    Args:
        tensor (Tensor): the input tensor

    For example:

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> import oneflow as flow

        >>> tensor = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >>> # tensor on rank0
        >>> tensor # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> # tensor on rank1
        >>> tensor # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> flow.comm.all_reduce(tensor)
        >>> tensor.numpy()
        array([[3, 5],
               [7, 9]])

    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.device.index == flow.env.get_local_rank()
    assert tensor.is_local
    device_type = tensor.device.type
    placement = flow.env.all_device_placement(device_type)
    result = tensor.to_consistent(
        placement=placement, sbp=flow.sbp.partial_sum
    ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)

    tensor.data = result.to_local()


def all_gather(tensor_list, tensor):
    """
    Gathers tensors from the whole group in a list.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.

    For example:

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> import oneflow as flow

        >>> input = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >>> # input on rank0
        >>> input # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> # input on rank1
        >>> input # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> tensor_list = [flow.zeros(2, 2, dtype=flow.int64) for _ in range(2)]
        >>> flow.comm.all_gather(tensor_list, input)
        >>> # result on rank0
        >>> tensor_list # doctest: +ONLY_CHECK_RANK_0
        [tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:0', dtype=oneflow.int64)]
        >>> # result on rank1
        >>> tensor_list # doctest: +ONLY_CHECK_RANK_1
        [tensor([[1, 2],
                [3, 4]], device='cuda:1', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)]

    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert isinstance(tensor_list, list)
    assert len(tensor_list) == flow.env.get_world_size()
    assert tensor.device.index == flow.env.get_local_rank()
    assert tensor.is_local
    tensor = tensor.expand(*([1] + list(tensor.shape)))
    device_type = tensor.device.type
    placement = flow.env.all_device_placement(device_type)
    tensor = tensor.to_consistent(
        placement=placement, sbp=flow.sbp.split(0)
    ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)
    assert len(tensor_list) == flow.env.get_world_size()
    for i in range(tensor.shape[0]):
        tensor_list[i] = tensor[i].to_local()


def broadcast(tensor, src):
    """
    Broadcasts the tensor to the whole group.
    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> import oneflow as flow
        >>> tensor = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >>> # input on rank0
        >>> tensor # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> # input on rank1
        >>> tensor # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> flow.comm.broadcast(tensor, 0)
        >>> # result on rank0
        >>> tensor # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)

    """
    assert isinstance(src, int)
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.is_local
    flow._C.broadcast(tensor, src_rank=src, inplace=True)


def scatter(tensor, scatter_list=None, src=0):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Args:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank (default is 0)
    """
    assert isinstance(src, int)
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.is_local
    out_shape = tensor.shape
    if flow.env.get_rank() == src:
        tensor.data = scatter_list[src]
        assert isinstance(scatter_list, list)
        assert len(scatter_list) == flow.env.get_world_size()
        for i in range(len(scatter_list)):
            if i == src:
                continue
            assert isinstance(scatter_list[i], flow._oneflow_internal.Tensor)
            assert scatter_list[i].is_local
            assert (
                scatter_list[i].shape == out_shape
            ), f"invalid tensor size at index {i}: {out_shape} vs {scatter_list[i].shape}"
            flow.comm.send(scatter_list[i], i)
    # send/recv on the same rank is invalid
    if flow.env.get_rank() != src:
        flow.comm.recv(src, out=tensor)


def reduce(tensor, dst):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank

    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.is_local
    assert isinstance(dst, int)
    original_tensor = flow._C.identity(tensor)
    flow.comm.all_reduce(tensor)
    if flow.env.get_rank() != dst:
        tensor.data = original_tensor


def all_to_all(output_tensor_list, input_tensor_list):
    """
    Each process scatters list of input tensors to all processes in a group and
    return gathered list of tensors in output list.

    Args:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.

    """

    def _check_list(tensor_list):
        assert isinstance(tensor_list, list)
        assert len(tensor_list) == flow.env.get_world_size()
        shape = tensor_list[0].shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        for tensor in tensor_list:
            assert isinstance(tensor, flow._oneflow_internal.Tensor)
            assert tensor.is_local
            assert shape == tensor.shape
            assert dtype == tensor.dtype
            assert device == tensor.device

    _check_list(output_tensor_list)
    _check_list(input_tensor_list)

    assert input_tensor_list[0].shape == output_tensor_list[0].shape
    assert input_tensor_list[0].dtype == output_tensor_list[0].dtype
    assert input_tensor_list[0].device == output_tensor_list[0].device

    for i in range(flow.env.get_world_size()):
        flow.comm.scatter(
            output_tensor_list[i],
            input_tensor_list if i == flow.env.get_rank() else [],
            src=i,
        )


def barrier():
    """
    Synchronizes all processes.

    """
    flow._oneflow_internal.eager.multi_client.Sync()


def reduce_scatter(output, input_list):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.

    """
    assert isinstance(output, flow._oneflow_internal.Tensor)
    assert output.is_local
    assert isinstance(input_list, list)
    assert len(input_list) == flow.env.get_world_size()
    output_shape = output.shape
    device_type = output.device.type
    placement = flow.env.all_device_placement(device_type)
    reduced_tensor_list = []
    for tensor in input_list:
        assert tensor.is_local
        assert tensor.shape == output_shape
        tensor = tensor.to_consistent(
            placement=placement, sbp=flow.sbp.partial_sum
        ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        reduced_tensor_list.append(tensor.to_local())
    output.data = reduced_tensor_list[flow.env.get_rank()]


def gather(tensor, gather_list=None, dst=0):
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)

    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.is_local
    shape = tensor.shape
    dtype = tensor.dtype
    tensor = tensor.expand(*([1] + list(shape)))
    device_type = tensor.device.type
    placement = flow.env.all_device_placement(device_type)
    tensor = tensor.to_consistent(
        placement=placement, sbp=flow.sbp.split(0)
    ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)

    if gather_list is None:
        gather_list = [
            flow.empty(shape, dtype=dtype) for _ in range(flow.env.get_world_size())
        ]

    assert gather_list is not None
    assert isinstance(gather_list, list)
    assert len(gather_list) == flow.env.get_world_size()
    for i in range(tensor.shape[0]):
        gather_list[i] = tensor[i].to_local()
