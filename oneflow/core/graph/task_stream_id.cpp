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

StreamIndexGenerator* StreamIndexGeneratorManager::GetGenerator(const DeviceId& device_id) {
  std::unique_lock<std::mutex> lck(mtx_);
  auto iter = generators_.find(device_id);
  if (iter == generators_.end()) {
    iter = generators_.emplace(device_id, std::make_unique<StreamIndexGenerator>()).first;
  }
  return iter->second.get();
}

void TaskStreamIndexGetterRegistry::Register(const key_t& key, const stream_index_getter& getter) {
  bool insert_success = stream_index_getter_map_.emplace(key, getter).second;
  if (!insert_success) {
    std::cerr << "DeviceType " << key.first << ", TaskType " << key.second
              << " was already registered";
    abort();
  }
}

Maybe<StreamId::stream_index_t> TaskStreamIndexGetterRegistry::Dispatch(
    DeviceType device_type, TaskType task_type, StreamIndexGenerator* generator) {
  auto key = std::make_pair(device_type, task_type);
  auto it = stream_index_getter_map_.find(key);
  CHECK_OR_RETURN(it != stream_index_getter_map_.end())
      << "TaskType: " << key.first << ", DeviceType: " << key.second << " has not been registered";
  return it->second(generator);
}

StreamId::stream_index_t GetTaskStreamIndex(TaskType task_type, const DeviceId& device_id) {
  auto* generator = Global<StreamIndexGeneratorManager>::Get()->GetGenerator(device_id);
  auto stream_index = CHECK_JUST(TaskStreamIndexGetterRegistry::Instance().Dispatch(
      device_id.device_type(), task_type, generator));
  return stream_index;
}

StreamId::stream_index_t GetComputeTaskStreamIndex(DeviceType device_type,
                                                   StreamIndexGenerator* generator) {
  if (device_type == DeviceType::kCPU) {
    size_t cpu_device_num = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
    return generator->GenerateNamedRoundRobin("CPU_COMPUTE", cpu_device_num);
  } else {
    return generator->GenerateNamed("COMPUTE");
  }
}

StreamId GenerateComputeTaskStreamId(const DeviceId& device_id) {
  auto* stream_index_generator =
      Global<StreamIndexGeneratorManager>::Get()->GetGenerator(device_id);
  StreamId::stream_index_t stream_index =
      GetComputeTaskStreamIndex(device_id.device_type(), stream_index_generator);
  return StreamId{device_id, stream_index};
}

StreamId GenerateComputeTaskStreamId(int64_t rank, DeviceType device_type, int64_t device_index) {
  DeviceId device_id{static_cast<DeviceId::rank_t>(rank), device_type,
                     static_cast<DeviceId::device_index_t>(device_index)};
  return GenerateComputeTaskStreamId(device_id);
}

StreamId GenerateNamedTaskStreamId(const DeviceId& device_id, const std::string& name) {
  auto* stream_index_generator =
      Global<StreamIndexGeneratorManager>::Get()->GetGenerator(device_id);
  auto stream_index = stream_index_generator->GenerateNamed(name);
  return StreamId{device_id, stream_index};
}

StreamId GenerateNamedTaskStreamId(int64_t rank, DeviceType device_type, int64_t device_index,
                                   const std::string& name) {
  DeviceId device_id{static_cast<DeviceId::rank_t>(rank), device_type,
                     static_cast<DeviceId::device_index_t>(device_index)};
  return GenerateNamedTaskStreamId(device_id, name);
}

}  // namespace oneflow
