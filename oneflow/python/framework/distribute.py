from __future__ import absolute_import

from contextlib import contextmanager

import oneflow.python.framework.distribute_context as distribute_ctx
from oneflow.python.oneflow_export import oneflow_export


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
def deprecated_mirrored_strategy():
    print(
        "WARNING:",
        "/".join(deprecated_mirrored_strategy._ONEFLOW_API),
        "will be removed in the future, use oneflow.scope.mirrored_view instead.",
    )
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
def MirroredStrategyEnabled() -> bool:
    """Determines whether mirroed strategy is enable in current context where this function is called.

    Returns:
        bool: `True` if mirrored strategy is enabled, otherwise `False`.
    
    """
    return distribute_ctx.IsMirroredStrategyEnabled()


@oneflow_export("distribute.consistent_strategy")
def deprecated_consistent_strategy():
    print(
        "WARNING:",
        "/".join(deprecated_consistent_strategy._ONEFLOW_API),
        "will be removed in the future, use oneflow.scope.consistent_view instead.",
    )
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
def ConsistentStrategyEnabled() -> bool:
    """Determines whether consistent strategy is enable in current context where this function is called.

    Returns:
        bool: `True` if consistent strategy is enabled, otherwise `False`.
    
    """
    return distribute_ctx.IsConsistentStrategyEnabled()


@oneflow_export("distribute.split")
def split(axis: int) -> SplitDistribute:
    """Generate a split scheme in which op will be splitted at `axis`.

    Args:
        axis (int): At `axis` the op will be splitted. 

    Returns:
        SplitDistribute: Split scheme object, often required by `with_distribute` method of `Blob` or `oneflow.get_variable`.
    
    Example::

        weight = weight.with_distribute(distribute.split(1))

    """
    assert type(axis) is int
    assert str(axis) in _axis_str2split_axis_obj, "not a valid split. expected: [0, 11)"
    return _axis_str2split_axis_obj[str(axis)]


@oneflow_export("distribute.broadcast")
def broadcast() -> BroadcastDistribute:
    """Generate a broadcast scheme.

    Returns:
        BroadcastDistribute: Broadcast scheme object, often required by `with_distribute` method of `Blob` or `oneflow.get_variable`.
    
    Example::

        segment_ids = segment_ids.with_distribute(flow.distribute.broadcast())
    
    """
    return _broadcast


@oneflow_export("distribute.auto")
def auto() -> AutoDistribute:
    """Generate a broadcast scheme.

    Returns:
        AutoDistribute: Auto distribute scheme object, often required by `with_distribute` method of `Blob` or `oneflow.get_variable`.
    
    """
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
