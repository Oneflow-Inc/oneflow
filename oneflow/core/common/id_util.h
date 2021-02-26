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
#include "oneflow/core/common/device_type.pb.h"
#include <climits>

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// | node_index | process_index |                                                         |
// | --- 12 --- | ----- 7 ----- |                                                         |
// |         ProcessId          | device_type | device_index  |                           |
// | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |                           |
// |                        DeviceId                          | stream_index |            |
// | ------------------------- 31 --------------------------- | ---- 12 ---- |            |
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

using node_index_t = uint32_t;
using process_index_t = uint32_t;
using device_index_t = uint32_t;
using stream_index_t = uint32_t;
using task_index_t = uint32_t;
constexpr int kCPUDeviceIndex = 0;

class ProcessId {
 public:
  constexpr static int kNodeIndexBits = 12;
  constexpr static int kProcessIndexBits = 7;
  constexpr static node_index_t kMaxNodeIndex =
      (static_cast<node_index_t>(1) << kNodeIndexBits) - 1;
  constexpr static process_index_t kMaxProcessIndex =
      (static_cast<process_index_t>(1) << kProcessIndexBits) - 1;

  ProcessId(node_index_t node_index, process_index_t process_index)
      : node_index_(node_index), process_index_(process_index) {
    CHECK_LE(node_index, kMaxNodeIndex);
    CHECK_LE(process_index, kMaxProcessIndex);
  }
  node_index_t node_index() const { return node_index_; }
  process_index_t process_index() const { return process_index_; }
  bool operator==(const ProcessId& rhs) const {
    return node_index_ == rhs.node_index_ && process_index_ == rhs.process_index_;
  }
  bool operator!=(const ProcessId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    return std::hash<node_index_t>{}(node_index_) ^ std::hash<process_index_t>{}(process_index_);
  }

 private:
  node_index_t node_index_;
  process_index_t process_index_;
};

class DeviceId {
 public:
  constexpr static int kDeviceTypeBits = 5;
  constexpr static int kDeviceIndexBits = 7;
  constexpr static int kMaxDeviceTypeVal = (static_cast<int>(1) << kDeviceTypeBits) - 1;
  constexpr static device_index_t kMaxDeviceIndex =
      (static_cast<device_index_t>(1) << kDeviceIndexBits) - 1;

  DeviceId(const ProcessId& process_id, DeviceType device_type, device_index_t device_index)
      : process_id_(process_id), device_type_(device_type), device_index_(device_index) {
    CHECK_LE(static_cast<int>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }
  const ProcessId& process_id() const { return process_id_; }
  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }
  bool operator==(const DeviceId& rhs) const {
    return process_id_ == rhs.process_id_ && device_type_ == rhs.device_type_
           && device_index_ == rhs.device_index_;
  }
  bool operator!=(const DeviceId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    return process_id_.hash() ^ std::hash<int>{}(static_cast<int>(device_type_))
           ^ std::hash<device_index_t>{}(device_index_);
  }

 private:
  ProcessId process_id_;
  DeviceType device_type_;
  device_index_t device_index_;
};

class StreamId {
 public:
  constexpr static int kStreamIndexBits = 12;
  constexpr static stream_index_t kMaxStreamIndex =
      (static_cast<stream_index_t>(1) << kStreamIndexBits) - 1;

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
  size_t hash() const { return device_id_.hash() ^ std::hash<stream_index_t>{}(stream_index_); }

 private:
  DeviceId device_id_;
  stream_index_t stream_index_;
};

class TaskId {
 public:
  const static int kTaskIndexBits = 21;
  constexpr static task_index_t kMaxTaskIndex =
      (static_cast<task_index_t>(1) << kTaskIndexBits) - 1;

  TaskId(const StreamId& stream_id, task_index_t task_index)
      : stream_id_(stream_id), task_index_(task_index) {
    CHECK_LE(task_index_, kMaxTaskIndex);
  }
  // TaskId(uint64_t global_stream_index, uint32_t task_index);
  const StreamId& stream_id() const { return stream_id_; }
  // uint64_t global_stream_index() const;
  task_index_t task_index() const { return task_index_; }
  bool operator==(const TaskId& rhs) const {
    return stream_id_ == rhs.stream_id_ && task_index_ == rhs.task_index_;
  }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }
  size_t hash() const { return stream_id_.hash() ^ std::hash<task_index_t>{}(task_index_); }

 private:
  StreamId stream_id_;
  task_index_t task_index_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::ProcessId> {
  size_t operator()(const oneflow::ProcessId& process_id) const { return process_id.hash(); }
};

template<>
struct hash<oneflow::DeviceId> {
  size_t operator()(const oneflow::DeviceId& device_id) const { return device_id.hash(); }
};

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
