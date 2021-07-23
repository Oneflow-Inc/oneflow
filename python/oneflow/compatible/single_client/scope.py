from oneflow.compatible.single_client.experimental.name_scope import name_scope
from oneflow.compatible.single_client.framework.scope_util import (
    deprecated_current_scope,
)
from oneflow.compatible.single_client.framework.placement_util import api_placement
from oneflow.compatible.single_client.framework.distribute import (
    DistributeMirroredStrategy,
)
from oneflow.compatible.single_client.framework.distribute import (
    MirroredStrategyEnabled,
)
from oneflow.compatible.single_client.framework.distribute import (
    DistributeConsistentStrategy,
)
from oneflow.compatible.single_client.framework.distribute import (
    ConsistentStrategyEnabled,
)
