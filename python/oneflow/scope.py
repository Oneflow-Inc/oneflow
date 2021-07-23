from oneflow.experimental.name_scope import name_scope as namespace
from oneflow.framework.distribute import (
    ConsistentStrategyEnabled as consistent_view_enabled,
)
from oneflow.framework.distribute import DistributeConsistentStrategy as consistent_view
from oneflow.framework.distribute import DistributeMirroredStrategy as mirrored_view
from oneflow.framework.distribute import (
    MirroredStrategyEnabled as mirrored_view_enabled,
)
from oneflow.framework.placement_util import api_placement as placement
from oneflow.framework.scope_util import deprecated_current_scope as current_scope
