/*
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
*/
/*
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
*/
#include "oneflow/core/graph/stream_index_getter_registry.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/stream_index_getter_registry_manager.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

StreamIndexGetterRegistry& StreamIndexGetterRegistry::SetStreamIndexGetterFn(
    StreamIndexGetterFn func) {
  StreamIndexGetterRegistryManager::Get().StreamIndexGetterFuncs()[dev_task_type_] = func;
  return *this;
}

}  // namespace oneflow