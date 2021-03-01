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
#include "oneflow/core/graph/stream_index_getter_registry_manager.h"

namespace oneflow {

StreamIndexGetterRegistryManager& StreamIndexGetterRegistryManager::Get() {
  static StreamIndexGetterRegistryManager mgr;
  return mgr;
}

StreamIndexKeyMap<StreamIndexGetterFn>& StreamIndexGetterRegistryManager::StreamIndexGetterFuncs() {
  return stream_index_getter_funcs_;
}

StreamIndexGetterFn StreamIndexGetterRegistryManager::GetStreamIndexGetterFunc(DeviceType dev_type,
                                                                               TaskType task_type) {
  auto strm_idx_getter = StreamIndexGetterRegistryManager::Get().StreamIndexGetterFuncs();
  std::pair<DeviceType, TaskType> dev_task_type(dev_type, task_type);
  if (strm_idx_getter.find(dev_task_type) == strm_idx_getter.end()) { UNIMPLEMENTED(); }

  return strm_idx_getter[dev_task_type];
}

}  // namespace oneflow