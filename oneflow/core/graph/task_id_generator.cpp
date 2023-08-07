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
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/graph/task_id.h"
#include "oneflow/core/graph/task_id_generator.h"

namespace oneflow {

void TaskIdGenerator::GetTaskIndex(HashMap<int64_t, uint32_t>* task_index_state) {
  for (const auto& pair : stream_id2task_index_counter_) {
    const int64_t i64_stream_id = EncodeStreamIdToInt64(pair.first);
    (*task_index_state)[i64_stream_id] = pair.second;
  }
}

void TaskIdGenerator::TryUpdateTaskIndex(const HashMap<int64_t, uint32_t>& task_index_state) {
  for (auto& pair : stream_id2task_index_counter_) {
    const int64_t i64_stream_id = EncodeStreamIdToInt64(pair.first);
    uint32_t initial_task_index = 0;
    if (task_index_state.count(i64_stream_id) != 0) {
      initial_task_index = task_index_state.at(i64_stream_id);
    }
    pair.second = std::max(pair.second, initial_task_index);
  }

  // try update the task_index_init_state
  for (const auto& pair : task_index_state) {
    const auto& key = pair.first;
    const auto& val = pair.second;
    if (task_index_init_state_.count(key) != 0) {
      task_index_init_state_[key] = std::max(task_index_init_state_.at(key), val);
    } else {
      task_index_init_state_[key] = val;
    }
  }
}

TaskId TaskIdGenerator::Generate(const StreamId& stream_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (stream_id2task_index_counter_.count(stream_id) == 0) {
    uint32_t init_task_index = 0;
    const int64_t i64_stream_id = EncodeStreamIdToInt64(stream_id);
    if (task_index_init_state_.count(i64_stream_id) != 0) {
      init_task_index = task_index_init_state_.at(i64_stream_id);
    }
    stream_id2task_index_counter_[stream_id] = init_task_index;
  }
  task_index_t task_index = stream_id2task_index_counter_[stream_id]++;
  return TaskId{stream_id, task_index};
}

}  // namespace oneflow
