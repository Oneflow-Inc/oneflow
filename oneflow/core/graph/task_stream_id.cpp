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
#include "oneflow/core/graph/task_stream_id.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

std::unique_ptr<StreamIndexGeneratorManager>& StreamIndexGeneratorManager::Ptr(
    bool create_when_absent) {
  thread_local std::unique_ptr<StreamIndexGeneratorManager> ptr;
  if (!ptr && create_when_absent) { ptr.reset(new StreamIndexGeneratorManager()); }
  return ptr;
}

StreamIndexGenerator* StreamIndexGeneratorManager::GetGenerator(const DeviceId& device_id) {
  std::unique_lock<std::mutex> lck(mtx_);
  auto iter = generators_.find(device_id);
  if (iter == generators_.end()) {
    iter = generators_.emplace(device_id, std::make_unique<StreamIndexGenerator>()).first;
  }
  return iter->second.get();
}

void TaskStreamIndexFactory::RegisterGetter(const key_t& key,
                                            const stream_index_getter_fn& getter) {
  bool insert_success = stream_index_getter_map_.emplace(key, getter).second;
  if (!insert_success) {
    std::cerr << "DeviceType " << key.first << ", TaskType " << key.second
              << " was already registered";
    abort();
  }
}

Maybe<StreamId::stream_index_t> TaskStreamIndexFactory::GetStreamIndex(TaskType task_type,
                                                                       const DeviceId& device_id) {
  auto key = std::make_pair(device_id.device_type(), task_type);
  auto it = stream_index_getter_map_.find(key);
  CHECK_OR_RETURN(it != stream_index_getter_map_.end())
      << "DeviceType " << key.first << ", TaskType " << key.second << " has not been registered";
  const stream_index_getter_fn& getter = it->second;
  return getter(device_id);
}

Maybe<StreamId::stream_index_t> GetTaskStreamIndex(TaskType task_type, const DeviceId& device_id) {
  return TaskStreamIndexFactory::Instance().GetStreamIndex(task_type, device_id);
}

StreamId GenerateComputeTaskStreamId(const DeviceId& device_id) {
  auto* stream_index_generator = StreamIndexGeneratorManager::Instance().GetGenerator(device_id);
  StreamId::stream_index_t stream_index = 0;
  if (device_id.device_type() == DeviceType::kCPU) {
    size_t cpu_device_num = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
    stream_index = stream_index_generator->Generate("cpu_compute", cpu_device_num);
  } else {
    stream_index = stream_index_generator->Generate("compute");
  }
  return StreamId{device_id, stream_index};
}

StreamId GenerateComputeTaskStreamId(int64_t rank, DeviceType device_type, int64_t device_index) {
  DeviceId device_id{static_cast<DeviceId::rank_t>(rank), device_type,
                     static_cast<DeviceId::device_index_t>(device_index)};
  return GenerateComputeTaskStreamId(device_id);
}

StreamId GenerateNamedTaskStreamId(const DeviceId& device_id, const std::string& name) {
  auto* stream_index_generator = StreamIndexGeneratorManager::Instance().GetGenerator(device_id);
  auto stream_index = stream_index_generator->Generate(name);
  return StreamId{device_id, stream_index};
}

StreamId GenerateNamedTaskStreamId(int64_t rank, DeviceType device_type, int64_t device_index,
                                   const std::string& name) {
  DeviceId device_id{static_cast<DeviceId::rank_t>(rank), device_type,
                     static_cast<DeviceId::device_index_t>(device_index)};
  return GenerateNamedTaskStreamId(device_id, name);
}

}  // namespace oneflow
