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
from __future__ import absolute_import

from contextlib import contextmanager

import oneflow.python.framework.distribute_context as distribute_ctx
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate
import traceback


class Distribute(object):
    def __init__(self):
        pass


class AutoDistribute(Distribute):
    def __init__(self):
        Distribute.__init__(self)


class BroadcastDistribute(Distribute):
    def __init__(self):
        Distribute.__init__(self)


class SplitDistribute(Distribute):
    def __init__(self, axis):
        Distribute.__init__(self)
        self.axis_ = axis

    @property
    def axis(self):
        return self.axis_


@oneflow_export("distribute.mirrored_strategy")
@oneflow_deprecate()
def deprecated_mirrored_strategy():
    print(
        "WARNING:",
        "oneflow.distribute.mirrored_strategy",
        "will be removed in the future, use {} instead.".format(
            "oneflow.scope.mirrored_view"
        ),
    )
    print(traceback.format_stack()[-2])
    return DistributeMirroredStrategy()


@oneflow_export("scope.mirrored_view")
class DistributeMirroredStrategy(distribute_ctx.DistributeStrategy):
    r"""Create a scope in mirrored view. All operators within the scope will be mirrored among diffierent accelerators.
    Usage::

        with oneflow.scope.mirrored_view():
            ...

    """

    def __init__(self):
        distribute_ctx.DistributeStrategy.__init__(self, True)


@oneflow_export("distribute.mirrored_strategy_enabled")
@oneflow_deprecate()
def deprecated_mirrored_strategy_enabled():
    print(
        "WARNING:",
        "oneflow.distribute.mirrored_strategy_enabled",
        "will be removed in the future, use {} instead.".format(
            "oneflow.scope.mirrored_view_enabled"
        ),
    )
    print(traceback.format_stack()[-2])
    return MirroredStrategyEnabled()


@oneflow_export("scope.mirrored_view_enabled")
def MirroredStrategyEnabled() -> bool:
    r"""

    Returns:
        bool: `True` if mirrored strategy is enabled in current context where this function is called.

    """
    return distribute_ctx.IsMirroredStrategyEnabled()


@oneflow_export("distribute.consistent_strategy")
@oneflow_deprecate()
def deprecated_consistent_strategy():
    print(
        "WARNING:",
        "oneflow.distribute.consistent_strategy",
        "will be removed in the future, use {} instead.".format(
            "oneflow.scope.consistent_view"
        ),
    )
    print(traceback.format_stack()[-2])
    return DistributeConsistentStrategy()


@oneflow_export("scope.consistent_view")
class DistributeConsistentStrategy(distribute_ctx.DistributeStrategy):
    r"""Create a scope in consistent view. All operators within the scope will be automatically parallelized among diffierent accelerators for best performance and least data transfer.
    Usage::

        with oneflow.scope.consistent_view():
            ...

    """

    def __init__(self):
        distribute_ctx.DistributeStrategy.__init__(self, False)


@oneflow_export("distribute.consistent_strategy_enabled")
@oneflow_deprecate()
def deprecated_consistent_strategy_enabled():
    print(
        "WARNING:",
        "oneflow.distribute.consistent_strategy_enabled",
        "will be removed in the future, use {} instead.".format(
            "oneflow.scope.consistent_view_enabled"
        ),
    )
    print(traceback.format_stack()[-2])
    return ConsistentStrategyEnabled()


@oneflow_export("scope.consistent_view_enabled")
def ConsistentStrategyEnabled() -> bool:
    r"""

    Returns:
        bool: `True` if consistent strategy is enabled in current context where this function is called.

    """
    return distribute_ctx.IsConsistentStrategyEnabled()


@oneflow_export("distribute.split")
def split(axis: int) -> SplitDistribute:
    assert type(axis) is int
    assert str(axis) in _axis_str2split_axis_obj, "not a valid split. expected: [0, 11)"
    return _axis_str2split_axis_obj[str(axis)]


@oneflow_export("distribute.broadcast")
def broadcast() -> BroadcastDistribute:
    return _broadcast


@oneflow_export("distribute.auto")
def auto() -> AutoDistribute:
    return _auto


@oneflow_export("distribute.assert_is_valid_distribute")
def assert_is_valid_distribute(distribute: Distribute) -> None:
    assert isinstance(
        distribute, Distribute
    ), """not a valid distribute policy.
           expected: 1) oneflow.distribute.split(axis); 2) oneflow.distribute.broadcast(); 3) oneflow.distribute.auto()"""


_auto = AutoDistribute()
_broadcast = BroadcastDistribute()
_axis_str2split_axis_obj = dict()
for i in range(11):
    class_name = "Split_Axis%d" % i
    _axis_str2split_axis_obj[str(i)] = SplitDistribute(i)
