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
from oneflow.compatible.single_client.experimental.namescope import (
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
