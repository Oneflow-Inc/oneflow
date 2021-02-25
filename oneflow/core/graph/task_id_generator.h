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
#ifndef ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_
#define ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_util.h"

namespace oneflow {

class TaskIdGenerator final {
 public:
  TaskIdGenerator() = default;
  OF_DISALLOW_COPY_AND_MOVE(TaskIdGenerator);
  ~TaskIdGenerator() = default;

  TaskId Generate(ProcessId process_id, StreamId stream_id);

 private:
  using process_stream_key_t = std::pair<ProcessId, StreamId>;
  HashMap<process_stream_key_t, uint32_t> process_stream2task_index_counter_;
};

inline TaskId TaskIdGenerator::Generate(ProcessId process_id, StreamId stream_id) {
  process_stream_key_t key = std::make_pair(process_id, stream_id);
  uint32_t task_index = process_stream2task_index_counter_[key]++;
  return TaskId(process_id, stream_id, task_index);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_
