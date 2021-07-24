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
from oneflow import oneflow_deprecate
from oneflow.compatible.single_client.framework import (
    distribute_context as distribute_ctx,
)


@oneflow_deprecate()
def deprecated_mirrored_strategy():
    print(
        "WARNING:",
        "oneflow.compatible.single_client.distribute.mirrored_strategy",
        "will be removed in the future, use {} instead.".format(
            "oneflow.compatible.single_client.scope.mirrored_view"
        ),
    )
    print(traceback.format_stack()[-2])
    return DistributeMirroredStrategy()


class DistributeMirroredStrategy(distribute_ctx.DistributeStrategy):
    """Create a scope in mirrored view. All operators within the scope will be mirrored among diffierent accelerators.
    Usage::

        with oneflow.compatible.single_client.scope.mirrored_view():
            ...

    """

    def __init__(self):
        distribute_ctx.DistributeStrategy.__init__(self, True)


from oneflow import oneflow_deprecate


@oneflow_deprecate()
def deprecated_mirrored_strategy_enabled():
    print(
        "WARNING:",
        "oneflow.compatible.single_client.distribute.mirrored_strategy_enabled",
        "will be removed in the future, use {} instead.".format(
            "oneflow.compatible.single_client.scope.mirrored_view_enabled"
        ),
    )
    print(traceback.format_stack()[-2])
    return MirroredStrategyEnabled()


def MirroredStrategyEnabled() -> bool:
    """

    Returns:
        bool: `True` if mirrored strategy is enabled in current context where this function is called.

    """
    return distribute_ctx.IsMirroredStrategyEnabled()


from oneflow import oneflow_deprecate


@oneflow_deprecate()
def deprecated_consistent_strategy():
    print(
        "WARNING:",
        "oneflow.compatible.single_client.distribute.consistent_strategy",
        "will be removed in the future, use {} instead.".format(
            "oneflow.compatible.single_client.scope.consistent_view"
        ),
    )
    print(traceback.format_stack()[-2])
    return DistributeConsistentStrategy()


class DistributeConsistentStrategy(distribute_ctx.DistributeStrategy):
    """Create a scope in consistent view. All operators within the scope will be automatically parallelized among diffierent accelerators for best performance and least data transfer.

    Usage::

        with oneflow.compatible.single_client.scope.consistent_view():
            ...

    """

    def __init__(self):
        distribute_ctx.DistributeStrategy.__init__(self, False)


from oneflow import oneflow_deprecate


@oneflow_deprecate()
def deprecated_consistent_strategy_enabled():
    print(
        "WARNING:",
        "oneflow.compatible.single_client.distribute.consistent_strategy_enabled",
        "will be removed in the future, use {} instead.".format(
            "oneflow.compatible.single_client.scope.consistent_view_enabled"
        ),
    )
    print(traceback.format_stack()[-2])
    return ConsistentStrategyEnabled()


def ConsistentStrategyEnabled() -> bool:
    """

    Returns:
        bool: `True` if consistent strategy is enabled in current context where this function is called.

    """
    return distribute_ctx.IsConsistentStrategyEnabled()


def split(axis: int) -> oneflow._oneflow_internal.distribute.SplitDistribute:
    """Generate a split scheme in which op will be splitted at `axis`.

    Args:
        axis (int): At `axis` the op will be splitted.

    Returns:
        SplitDistribute: Split scheme object, often required by `with_distribute` method of `Blob` or `oneflow.compatible.single_client.get_variable`.

    Example::
        weight = weight.with_distribute(distribute.split(1))

    """
    assert type(axis) is int
    return oneflow._oneflow_internal.distribute.split(axis)


def broadcast() -> oneflow._oneflow_internal.distribute.BroadcastDistribute:
    """Generate a broadcast scheme.

    Returns:
        BroadcastDistribute: Broadcast scheme object, often required by `with_distribute` method of `Blob` or `oneflow.compatible.single_client.get_variable`.

    Example::
        segment_ids = segment_ids.with_distribute(flow.distribute.broadcast())

    """
    return oneflow._oneflow_internal.distribute.broadcast()


def auto() -> oneflow._oneflow_internal.distribute.AutoDistribute:
    """Generate a broadcast scheme.

    Returns:
        AutoDistribute: Auto distribute scheme object, often required by `with_distribute` method of `Blob` or `oneflow.compatible.single_client.get_variable`.

    """
    return oneflow._oneflow_internal.distribute.auto()


def assert_is_valid_distribute(
    distribute: oneflow._oneflow_internal.distribute.Distribute,
) -> None:
    assert isinstance(
        distribute, oneflow._oneflow_internal.distribute.Distribute
    ), "not a valid distribute policy.\n           expected: 1) oneflow.compatible.single_client.distribute.split(axis); 2) oneflow.compatible.single_client.distribute.broadcast(); 3) oneflow.compatible.single_client.distribute.auto()"


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


def split_sbp(
    axis: int,
) -> oneflow._oneflow_internal.oneflow.core.job.sbp_parallel.SbpParallel:
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
