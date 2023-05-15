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
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/id_state.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

StreamIndexGenerator* TaskStreamIndexManager::GetGenerator(const DeviceId& device_id) {
  std::unique_lock<std::mutex> lck(mtx_);
  auto iter = generators_.find(device_id);
  if (iter == generators_.end()) {
    uint32_t init_stream_index = 0;
    const int64_t i64_device_id = EncodeDeviceIdToInt64(device_id);
    if (stream_index_init_state_.count(i64_device_id) != 0) {
      init_stream_index = stream_index_init_state_.at(i64_device_id);
    }
    iter = generators_.emplace(device_id, std::make_unique<StreamIndexGenerator>(init_stream_index))
               .first;
  }
  return iter->second.get();
}

TaskStreamIndexManager::stream_index_t TaskStreamIndexManager::GetTaskStreamIndex(
    TaskType task_type, const DeviceId& device_id) {
  auto* generator = GetGenerator(device_id);
  auto stream_index = CHECK_JUST(TaskStreamIndexGetterRegistry::Instance().Dispatch(
      device_id.device_type(), task_type, generator));
  return stream_index;
}

TaskStreamIndexManager::stream_index_t TaskStreamIndexManager::GetComputeTaskStreamIndex(
    const DeviceId& device_id) {
  auto* generator = GetGenerator(device_id);
  return GenerateComputeTaskStreamIndex(device_id.device_type(), generator);
}

TaskStreamIndexManager::stream_index_t TaskStreamIndexManager::GetNamedTaskStreamIndex(
    const DeviceId& device_id, const std::string& name) {
  auto* generator = GetGenerator(device_id);
  return generator->GenerateNamed(name);
}

void TaskStreamIndexManager::GetTaskStreamIndex(HashMap<int64_t, uint32_t>* stream_index_state) {
  for (auto& pair : generators_) {
    const int64_t i64_device_id = EncodeDeviceIdToInt64(pair.first);
    (*stream_index_state)[i64_device_id] = pair.second->GetCurrStreamIndex();
  }
}

void TaskStreamIndexManager::TryUpdateTaskStreamIndex(
    const HashMap<int64_t, uint32_t>& stream_index_state) {
  // Try Update generator's new_stream_index
  for (auto& pair : generators_) {
    const int64_t i64_device_id = EncodeDeviceIdToInt64(pair.first);
    uint32_t initial_stream_index = 0;
    if (stream_index_state.count(i64_device_id) != 0) {
      initial_stream_index = stream_index_state.at(i64_device_id);
    }
    pair.second->TryUpdateNextStreamIndex(initial_stream_index);
  }

  // try update stream_index_init_state
  for (const auto& pair : stream_index_state) {
    const auto& key = pair.first;
    const auto& val = pair.second;
    if (stream_index_init_state_.count(key) != 0) {
      stream_index_init_state_[key] = std::max(stream_index_init_state_.at(key), val);
    } else {
      stream_index_init_state_[key] = val;
    }
  }
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
      << "TaskType: " << key.second << ", DeviceType: " << key.first << " has not been registered";
  return it->second(generator);
}

StreamId::stream_index_t GenerateComputeTaskStreamIndex(DeviceType device_type,
                                                        StreamIndexGenerator* generator) {
  if (device_type == DeviceType::kCPU) {
    size_t cpu_device_num = Singleton<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
    return generator->GenerateNamedRoundRobin("CPU_COMPUTE", cpu_device_num);
  } else {
    return generator->GenerateNamed("COMPUTE");
  }
}

}  // namespace oneflow
