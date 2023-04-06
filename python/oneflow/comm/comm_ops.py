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
               [7, 9]], dtype=int64)

    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.device.index == flow.env.get_local_rank()
    assert tensor.is_local
    flow._C.local_all_reduce(tensor, inplace=True)


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
    placement = flow.placement.all(device_type)
    tensor = (
        tensor.to_global(placement=placement, sbp=flow.sbp.split(0))
        .to_global(placement=placement, sbp=flow.sbp.broadcast)
        .to_local()
    )
    assert len(tensor_list) == flow.env.get_world_size()
    # TODO(): getitem has bug on global tensor with size = [2, 1].
    for i in range(tensor.shape[0]):
        tensor_list[i] = tensor[i]


def all_gather_into_tensor(output_tensor, input_tensor):
    """
    Gather tensors from all ranks and put them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``oneflow.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``oneflow.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
            The input tensors in this API must have the same size across all ranks.

    For example:

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> # All tensors below are of flow.int64 dtype and on CUDA devices.
        >>> import oneflow as flow
        >>> tensor_in = flow.tensor([[1, 2, 3], [4, 5, 6]], dtype=flow.int64, device="cuda") + flow.env.get_rank() * 6
        >>> tensor_in # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2, 3],
                [4, 5, 6]], device='cuda:0', dtype=oneflow.int64)
        >>> # Output in concatenation form
        >>> tensor_out = flow.zeros(4, 3, dtype=flow.int64, device="cuda")
        >>> flow.comm.all_gather_into_tensor(tensor_out, tensor_in)
        >>> # result on rank0
        >>> tensor_out # doctest: +ONLY_CHECK_RANK_0
        tensor([[ 1,  2,  3],
                [ 4,  5,  6],
                [ 7,  8,  9],
                [10, 11, 12]], device='cuda:0', dtype=oneflow.int64)
        >>> # result on rank1
        >>> tensor_out # doctest: +ONLY_CHECK_RANK_1
        tensor([[ 1,  2,  3],
                [ 4,  5,  6],
                [ 7,  8,  9],
                [10, 11, 12]], device='cuda:1', dtype=oneflow.int64)
        >>> # Output in stack form
        >>> tensor_out2 = flow.zeros(2, 3, 2, dtype=flow.int64, device="cuda")
        >>> flow.comm.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> # result on rank0
        >>> tensor_out2 # doctest: +ONLY_CHECK_RANK_0
        tensor([[[ 1,  2],
                 [ 3,  4],
                 [ 5,  6]],
        <BLANKLINE>
                [[ 7,  8],
                 [ 9, 10],
                 [11, 12]]], device='cuda:0', dtype=oneflow.int64)
        >>> # result on rank1
        >>> tensor_out2 # doctest: +ONLY_CHECK_RANK_1
        tensor([[[ 1,  2],
                 [ 3,  4],
                 [ 5,  6]],
        <BLANKLINE>
                [[ 7,  8],
                 [ 9, 10],
                 [11, 12]]], device='cuda:1', dtype=oneflow.int64)

    """
    assert output_tensor.is_local
    assert input_tensor.is_local
    flow._C.local_all_gather(output_tensor, input_tensor)


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
    flow._C.comm_broadcast(tensor, src_rank=src, inplace=True)


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
    flow._oneflow_internal.eager.ClusterSync()


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
    placement = flow.placement.all(device_type)
    reduced_tensor_list = []
    for tensor in input_list:
        assert tensor.is_local
        assert tensor.shape == output_shape
        tensor = tensor.to_global(
            placement=placement, sbp=flow.sbp.partial_sum
        ).to_global(placement=placement, sbp=flow.sbp.broadcast)
        reduced_tensor_list.append(tensor.to_local())
    output.data = reduced_tensor_list[flow.env.get_rank()]


def reduce_scatter_tensor(output_tensor, input_tensor):
    """
    Reduces, then scatters a tensor to all ranks.

    Args:
        output (Tensor): Output tensor. It should have the same size across all
            ranks.
        input (Tensor): Input tensor to be reduced and scattered. Its size
            should be output tensor size times the world size. The input tensor
            can have one of the following shapes:
            (i) a concatenation of the output tensors along the primary
            dimension, or
            (ii) a stack of the output tensors along the primary dimension.
            For definition of "concatenation", see ``oneflow.cat()``.
            For definition of "stack", see ``oneflow.stack()``.

    For example:

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> # All tensors below are of flow.int64 dtype and on CUDA devices.
        >>> import oneflow as flow
        >>> tensor_in = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=flow.int64, device="cuda")
        >>> tensor_in # doctest: +ONLY_CHECK_RANK_0
        tensor([[ 1,  2,  3],
                [ 4,  5,  6],
                [ 7,  8,  9],
                [10, 11, 12]], device='cuda:0', dtype=oneflow.int64)
        >>> # Output in concatenation form
        >>> tensor_out = flow.zeros(2, 3, dtype=flow.int64, device="cuda")
        >>> flow.comm.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> # result on rank0
        >>> tensor_out # doctest: +ONLY_CHECK_RANK_0
        tensor([[ 2,  4,  6],
                [ 8, 10, 12]], device='cuda:0', dtype=oneflow.int64)
        >>> # result on rank1
        >>> tensor_out # doctest: +ONLY_CHECK_RANK_1
        tensor([[14, 16, 18],
                [20, 22, 24]], device='cuda:1', dtype=oneflow.int64)
        >>> # Output in stack form
        >>> tensor_in2 = tensor_in.reshape(2, 3, 2)
        >>> tensor_out2 = flow.zeros(2, 3, dtype=flow.int64, device="cuda")
        >>> flow.comm.reduce_scatter_tensor(tensor_out2, tensor_in2)
        >>> # result on rank0
        >>> tensor_out2 # doctest: +ONLY_CHECK_RANK_0
        tensor([[ 2,  4,  6],
                [ 8, 10, 12]], device='cuda:0', dtype=oneflow.int64)
        >>> # result on rank1
        >>> tensor_out2 # doctest: +ONLY_CHECK_RANK_1
        tensor([[14, 16, 18],
                [20, 22, 24]], device='cuda:1', dtype=oneflow.int64)

    """
    assert output_tensor.is_local
    assert input_tensor.is_local
    flow._C.local_reduce_scatter(output_tensor, input_tensor)


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
    placement = flow.placement.all(device_type)
    tensor = tensor.to_global(placement=placement, sbp=flow.sbp.split(0)).to_global(
        placement=placement, sbp=flow.sbp.broadcast
    )

    if gather_list is None:
        gather_list = [
            flow.empty(shape, dtype=dtype) for _ in range(flow.env.get_world_size())
        ]

    assert gather_list is not None
    assert isinstance(gather_list, list)
    assert len(gather_list) == flow.env.get_world_size()
    for i in range(tensor.shape[0]):
        gather_list[i] = tensor[i].to_local()
