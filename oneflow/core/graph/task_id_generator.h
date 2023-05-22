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

#include "oneflow/core/graph/task_id.h"
#include "oneflow/core/job/id_state.h"

namespace oneflow {

class TaskIdGenerator final {
 public:
  using task_index_t = TaskId::task_index_t;

  TaskIdGenerator() = default;
  OF_DISALLOW_COPY_AND_MOVE(TaskIdGenerator);
  ~TaskIdGenerator() = default;

  TaskId Generate(const StreamId& stream_id);

  void GetTaskIndex(HashMap<int64_t, uint32_t>* task_index_state);
  void TryUpdateTaskIndex(const HashMap<int64_t, uint32_t>& task_index_state);

 private:
  HashMap<StreamId, task_index_t> stream_id2task_index_counter_;
  // The task_index_init_state is used to initialize the `stream_id2task_index_counter_` hashmap.
  HashMap<int64_t, uint32_t> task_index_init_state_{};
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_
