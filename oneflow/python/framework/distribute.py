from __future__ import absolute_import

from contextlib import contextmanager

import oneflow.python.framework.distribute_context as distribute_ctx
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("distribute.mirrored_strategy")
class DistributeMirroredStrategy(distribute_ctx.DistributeStrategy):
    r"""Create a mirrored strategy scope. All operators within the scope will be mirrored among diffierent accelerators.
    Usage::

        with oneflow.distribute.mirrored_strategy():
            ...

    """

    def __enter__(self, *argc, **kwarg):
        distribute_ctx.PushMirroredStrategyEnabled(True)

    def __exit__(self, *argc, **kwarg):
        distribute_ctx.PopMirroredStrategyEnabled()


@oneflow_export("distribute.mirrored_strategy_enabled")
def MirroredStrategyEnabled():
    r"""

    Returns:
        bool: `True` if mirrored strategy is enabled in current context where this function is called.

    """
    return distribute_ctx.IsMirroredStrategyEnabled()


@oneflow_export("distribute.consistent_strategy")
class DistributeConsistentStrategy(distribute_ctx.DistributeStrategy):
    r"""Create a consistent strategy scope. All operators within the scope will be automatically parallelized among diffierent accelerators for best performance and least data transfer.
    Usage::

        with oneflow.distribute.consistent_strategy():
            ...

    """

    def __enter__(self, *argc, **kwarg):
        distribute_ctx.PushMirroredStrategyEnabled(False)

    def __exit__(self, *argc, **kwarg):
        distribute_ctx.PopMirroredStrategyEnabled()


@oneflow_export("distribute.consistent_strategy_enabled")
def ConsistentStrategyEnabled():
    r"""

    Returns:
        bool: `True` if consistent strategy is enabled in current context where this function is called.

    """
    return distribute_ctx.IsConsistentStrategyEnabled()


@oneflow_export("distribute.split")
def split(axis):
    assert type(axis) is int
    assert str(axis) in _axis_str2split_axis_obj, "not a valid split. expected: [0, 11)"
    return _axis_str2split_axis_obj[str(axis)]


@oneflow_export("distribute.broadcast")
def broadcast():
    return _broadcast


@oneflow_export("distribute.auto")
def auto():
    return _auto


@oneflow_export("distribute.assert_is_valid_distribute")
def assert_is_valid_distribute(distribute):
    assert isinstance(
        distribute, Distribute
    ), """not a valid distribute policy. 
           expected: 1) oneflow.distribute.split(axis); 2) oneflow.distribute.broadcast(); 3) oneflow.distribute.auto()"""


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


_auto = AutoDistribute()
_broadcast = BroadcastDistribute()
_axis_str2split_axis_obj = dict()
for i in range(11):
    class_name = "Split_Axis%d" % i
    _axis_str2split_axis_obj[str(i)] = SplitDistribute(i)
