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
#ifndef ONEFLOW_CORE_GRAPH_TASK_ID_H_
#define ONEFLOW_CORE_GRAPH_TASK_ID_H_

#include "oneflow/core/graph/stream_id.h"

namespace oneflow {

class TaskId {
 public:
  using task_index_t = uint32_t;

  const static size_t kTaskIndexBits = 21;
  constexpr static task_index_t kMaxTaskIndex =
      (task_index_t{1} << kTaskIndexBits) - task_index_t{1};

  TaskId(const StreamId& stream_id, task_index_t task_index)
      : stream_id_(stream_id), task_index_(task_index) {
    CHECK_LE(task_index_, kMaxTaskIndex);
  }

  const StreamId& stream_id() const { return stream_id_; }
  task_index_t task_index() const { return task_index_; }

  bool operator==(const TaskId& rhs) const {
    return stream_id_ == rhs.stream_id_ && task_index_ == rhs.task_index_;
  }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }

  size_t hash() const {
    size_t hash = stream_id_.hash();
    HashCombine(&hash, std::hash<task_index_t>{}(task_index_));
    return hash;
  }

 private:
  StreamId stream_id_;
  task_index_t task_index_;
};

int64_t EncodeTaskIdToInt64(const TaskId&);
TaskId DecodeTaskIdFromInt64(int64_t);

int64_t MachineId4ActorId(int64_t actor_id);
int64_t ThrdId4ActorId(int64_t actor_id);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::TaskId> {
  size_t operator()(const oneflow::TaskId& task_id) const { return task_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_GRAPH_TASK_ID_H_
