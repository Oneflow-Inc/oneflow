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
import traceback
from contextlib import contextmanager

import oneflow._oneflow_internal


def get_local_rank():
    return oneflow._oneflow_internal.GetLocalRank()


def get_rank():
    """Returns the rank of current process group.

    Returns:
        The rank of the process group.

    """
    return oneflow._oneflow_internal.GetRank()


def get_world_size():
    """Returns the number of processes in the current process group.

    Returns:
        The world size of the process group.

    """
    return oneflow._oneflow_internal.GetWorldSize()


def is_multi_client():
    return oneflow._oneflow_internal.IsMultiClient()


def split_sbp(axis: int) -> oneflow._oneflow_internal.sbp.sbp:
    """Generate a split scheme in which op will be splitted at `axis`.

    Args:
        axis (int): At `axis` the op will be splitted.

    Returns:
        SbpParallel: Split scheme object, often required by `to_consistent` method of `Tensor`

    Example::
        array = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        t1 = flow.tensor(array)
        ct2 = t1.to_consistent(sbp=flow.sbp.split(0), placement=("cuda", {0: [0, 1, 2, 3]}))

    """
    assert type(axis) is int
    return oneflow._oneflow_internal.sbp.split(axis)
