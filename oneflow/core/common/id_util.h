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
// | ----------------------------------- 64 bit ----------------------------------------- |
// | ----------- 19 ----------- | ------------------- 24 ------------------- | --- 21 --- |
// |         ProcessId          |                  StreamId                  |            |
// | --- 12 --- | ----- 7 ----- | ---- 5 ---- | ----- 7 ----- | ---- 12 ---- | --- 21 --- |
// | node_index | process_index | device_type | device_index  | stream_index | task_index |
// |                                      TaskId                                          |

class ProcessId {
 public:
  ProcessId(uint32_t node_index, uint32_t process_index);
  uint32_t node_index() const;
  uint32_t process_index() const;
  bool operator==(const ProcessId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const ProcessId& rhs) const { return !(*this == rhs); }

 private:
  using underlying_t = uint32_t;
  constexpr static int kNodeIndexBits = 12;
  constexpr static int kProcessIndexBits = 7;
  constexpr static int kBits = kNodeIndexBits + kProcessIndexBits;
  constexpr static int kFullBits = sizeof(underlying_t) * CHAR_BIT;
  static_assert(kBits <= kFullBits, "ProcessId bits layout is illegal");
  constexpr static int kReservedBits = kFullBits - kBits;

  friend class TaskId;
  friend class std::hash<ProcessId>;
  explicit ProcessId(underlying_t val) : val_(val) {}
  operator uint32_t() const { return val_; }

  underlying_t val_;
};

class DeviceId {
 public:
  DeviceId(DeviceType device_type, uint32_t device_index);
  DeviceType device_type() const;
  uint32_t device_index() const;
  bool operator==(const DeviceId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const DeviceId& rhs) const { return !(*this == rhs); }

 private:
  using underlying_t = uint32_t;
  constexpr static int kDeviceTypeBits = 5;
  constexpr static int kDeviceIndexBits = 7;
  constexpr static int kBits = kDeviceTypeBits + kDeviceIndexBits;
  constexpr static int kFullBits = sizeof(underlying_t) * CHAR_BIT;
  static_assert(kBits <= kFullBits, "DeviceId bits layout is illegal");
  constexpr static int kReservedBits = kFullBits - kBits;

  friend class StreamId;
  friend class std::hash<DeviceId>;
  explicit DeviceId(underlying_t val) : val_(val) {}
  operator underlying_t() const { return val_; }

  underlying_t val_;
};

class StreamId {
 public:
  StreamId(DeviceId device_id, uint32_t stream_index);
  StreamId(DeviceType device_type, uint32_t device_index, uint32_t stream_index);
  DeviceId device_id() const;
  DeviceType device_type() const;
  uint32_t device_index() const;
  uint32_t stream_index() const;
  bool operator==(const StreamId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const StreamId& rhs) const { return !(*this == rhs); }

 private:
  using underlying_t = uint32_t;
  constexpr static int kDeviceIdBits = DeviceId::kBits;
  constexpr static int kStreamIndexBits = 12;
  constexpr static int kBits = kDeviceIdBits + kStreamIndexBits;
  constexpr static int kFullBits = sizeof(underlying_t) * CHAR_BIT;
  static_assert(kBits <= kFullBits, "StreamId bits layout is illegal");
  constexpr static int kReservedBits = kFullBits - kBits;

  friend class TaskId;
  friend int64_t SerializeStreamIdToInt64(StreamId);
  friend StreamId DeserializeStreamIdFromInt64(int64_t);
  friend class std::hash<StreamId>;
  explicit StreamId(underlying_t val) : val_(val) {}
  operator underlying_t() const { return val_; }

  underlying_t val_;
};

class TaskId {
 public:
  TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index);
  TaskId(uint64_t global_stream_index, uint32_t task_index);
  ProcessId process_id() const;
  StreamId stream_id() const;
  uint64_t global_stream_index() const;
  uint32_t task_index() const;
  bool operator==(const TaskId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }

 private:
  using underlying_t = uint64_t;
  const static int kProcessIdBits = ProcessId::kBits;
  const static int kStreamIdBits = StreamId::kBits;
  const static int kTaskIndexBits = 21;
  const static int kBits = kProcessIdBits + kStreamIdBits + kTaskIndexBits;
  const static int kFullBits = sizeof(underlying_t) * CHAR_BIT;
  static_assert(kBits == kFullBits, "TaskId bits layout is illegal");

  friend int64_t SerializeTaskIdToInt64(TaskId);
  friend TaskId DeserializeTaskIdFromInt64(int64_t);
  friend class std::hash<TaskId>;
  explicit TaskId(underlying_t val) : val_(val) {}
  operator underlying_t() const { return val_; }

  underlying_t val_;
};

int64_t SerializeStreamIdToInt64(StreamId);
StreamId DeserializeStreamIdFromInt64(int64_t);

int64_t SerializeTaskIdToInt64(TaskId);
TaskId DeserializeTaskIdFromInt64(int64_t);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::ProcessId> {
  size_t operator()(const oneflow::ProcessId& process_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(process_id));
  }
};

template<>
struct hash<oneflow::DeviceId> {
  size_t operator()(const oneflow::DeviceId& device_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(device_id));
  }
};

template<>
struct hash<oneflow::StreamId> {
  size_t operator()(const oneflow::StreamId& stream_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(stream_id));
  }
};

template<>
struct hash<oneflow::TaskId> {
  size_t operator()(const oneflow::TaskId& task_id) const {
    return std::hash<uint64_t>{}(static_cast<uint64_t>(task_id));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
