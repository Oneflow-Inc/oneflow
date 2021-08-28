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

        >>> input = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.distributed.get_local_rank()
        >>> input # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> input # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> out = flow.comm.all_reduce(input)
        >>> out  # doctest: +ONLY_CHECK_RANK_0
        tensor([[3, 5],
                [7, 9]], device='cuda:0', dtype=oneflow.int64)
    """
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.device.index == flow.framework.distribute.get_local_rank()
    assert tensor.is_local
    placement = None
    machine_device_ids = {}
    nproc_per_node = int(
        flow.distributed.get_world_size() / flow.distributed.get_node_size()
    )
    for node_rank in range(flow.distributed.get_node_size()):
        machine_device_ids[node_rank] = range(nproc_per_node)
    placement = flow.placement(str(tensor.device).split(":")[0], machine_device_ids)

    assert placement != None
    tensor = tensor.to_consistent(
        placement=placement, sbp=flow.sbp.partial_sum
    ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)

    return tensor.to_local()
