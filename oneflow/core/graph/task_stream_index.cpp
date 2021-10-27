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
#include "oneflow/core/graph/task_stream_index.h"

namespace oneflow {

Maybe<void> TaskStreamIndexFactory::RegisterGetter(const key_t& key,
                                                   const stream_index_getter_fn& getter) {
  std::unique_lock<std::mutex> lck(mtx_);
  CHECK_OR_RETURN(stream_index_getter_map_.emplace(key, getter).second)
      << "DeviceType " << key.first << ", TaskType " << key.second << " was already registered";
  return Maybe<void>::Ok();
}

Maybe<StreamId::index_t> TaskStreamIndexFactory::Get(TaskType task_type,
                                                     const DeviceId& device_id) {
  std::unique_lock<std::mutex> lck(mtx_);
  auto key = std::make_pair(device_id.device_type(), task_type);
  auto it = stream_index_getter_map_.find(key);
  CHECK_OR_RETURN(it != stream_index_getter_map_.end())
      << "DeviceType " << key.first << ", TaskType " << key.second << " has not been registered";
  return it->second(device_id);
}

Maybe<StreamId::index_t> GetTaskStreamIndex(TaskType task_type, const DeviceId& device_id) {
  return TaskStreamIndexFactory::Instance().Get(task_type, device_id);
}

}  // namespace oneflow
