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
#ifndef ONEFLOW_CORE_COMMON_ID_UTIL_H_
#define ONEFLOW_CORE_COMMON_ID_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/device/device_id.h"

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// |            rank            | device_type | device_index  |                           |
// | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |                           |
// |                        DeviceId                          | stream_index |            |
// | ------------------------- 31 --------------------------- | ---- 12 ---- |            |
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

class StreamId {
 public:
  using stream_index_t = uint32_t;

  constexpr static size_t kStreamIndexBits = 12;
  constexpr static stream_index_t kMaxStreamIndex =
      (stream_index_t{1} << kStreamIndexBits) - stream_index_t{1};

  StreamId(const DeviceId& device_id, stream_index_t stream_index)
      : device_id_(device_id), stream_index_(stream_index) {
    CHECK_LE(stream_index, kMaxStreamIndex);
  }
  const DeviceId& device_id() const { return device_id_; }
  stream_index_t stream_index() const { return stream_index_; }
  bool operator==(const StreamId& rhs) const {
    return device_id_ == rhs.device_id_ && stream_index_ == rhs.stream_index_;
  }
  bool operator!=(const StreamId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = device_id_.hash();
    HashCombine(&hash, std::hash<stream_index_t>{}(stream_index_));
    return hash;
  }

 private:
  DeviceId device_id_;
  stream_index_t stream_index_;
};

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

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::StreamId> {
  size_t operator()(const oneflow::StreamId& stream_id) const { return stream_id.hash(); }
};

template<>
struct hash<oneflow::TaskId> {
  size_t operator()(const oneflow::TaskId& task_id) const { return task_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
