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
    assert isinstance(tensor, flow._oneflow_internal.Tensor)
    assert tensor.device.type == "cuda"
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
