from oneflow.compatible.single_client.experimental.name_scope import (
    name_scope as namespace,
)
from oneflow.compatible.single_client.framework.distribute import (
    ConsistentStrategyEnabled as consistent_view_enabled,
)
from oneflow.compatible.single_client.framework.distribute import (
    DistributeConsistentStrategy as consistent_view,
)
from oneflow.compatible.single_client.framework.distribute import (
    DistributeMirroredStrategy as mirrored_view,
)
from oneflow.compatible.single_client.framework.distribute import (
    MirroredStrategyEnabled as mirrored_view_enabled,
)
from oneflow.compatible.single_client.framework.placement_util import (
    api_placement as placement,
)
from oneflow.compatible.single_client.framework.scope_util import (
    deprecated_current_scope as current_scope,
)
