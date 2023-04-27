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
#include "oneflow/core/graph/task_id_generator.h"

namespace oneflow {

void TaskIdGenerator::SaveId() {
  for (const auto& pair : stream_id2task_index_counter_) {
    Singleton<MultiClientSessionContext>::Get()->GetIdStateMgr()->SetTaskIndexState(pair.first,
                                                                                    pair.second);
  }
}

inline TaskId TaskIdGenerator::Generate(const StreamId& stream_id) {
  if (stream_id2task_index_counter_.count(stream_id) == 0) {
    stream_id2task_index_counter_[stream_id] =
        Singleton<MultiClientSessionContext>::Get()->GetIdStateMgr()->GetTaskIndexState(stream_id);
  }
  task_index_t task_index = stream_id2task_index_counter_[stream_id]++;
  return TaskId{stream_id, task_index};
}

}  // namespace oneflow
