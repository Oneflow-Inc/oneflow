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
import oneflow.unittest


def all_reduce(tensor):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.
    After the call ``tensor`` is going to be bitwise identical in all processes.

    Args:
        input (Tensor): the input tensor

    For example:

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> import oneflow as flow

        >>> input = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >>> input # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> input # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> out = flow.comm.all_reduce(input)
        >>> out.numpy()
        array([[3, 5],
               [7, 9]])
    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.device.index == flow.env.get_local_rank()
    assert tensor.is_local
    device_type = tensor.device.type
    placement = flow.env.all_device_placement(device_type)
    tensor = tensor.to_consistent(
        placement=placement, sbp=flow.sbp.partial_sum
    ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)

    return tensor.to_local()


def all_gather(tensor_list, tensor):
    """
    Gathers tensors from the whole group in a list.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
        correctly-sized tensors to be used for output of the collective.

    For example:

    .. code-block:: python

        >>> # We have 1 process groups, 2 ranks.
        >>> import oneflow as flow

        >>> input = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >>> input # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> input # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> tensor_list = [flow.zeros(2, 2, dtype=flow.int64) for _ in range(2)]
        >>> flow.comm.all_gather(tensor_list, input)
        >>> tensor_list # doctest: +ONLY_CHECK_RANK_0
        [tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:0', dtype=oneflow.int64)]
        >>> tensor_list # doctest: +ONLY_CHECK_RANK_1
        [tensor([[1, 2],
                [3, 4]], device='cuda:1', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)]
    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.device.index == flow.env.get_local_rank()
    assert tensor.is_local
    tensor = tensor.expand([1] + list(tensor.shape))
    device_type = tensor.device.type
    tensor = tensor.to_consistent(
        placement=flow.env.all_device_placement(device_type), sbp=flow.sbp.split(0)
    )
    assert len(tensor_list) == tensor.shape[0]
    for i in range(len(tensor_list)):
        assert (
            tensor_list[i].shape == tensor[i].shape
        ), f"{tensor_list.shape} vs {tensor[i].shape}"
        assert (
            tensor_list[i].dtype == tensor[i].dtype
        ), f"{tensor_list[i].dtype} vs {tensor[i].dtype}"
        tensor_list[i] = tensor[i].to_local()
