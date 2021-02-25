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
#include <limits>

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

  // remove type cast operator in future
  operator uint32_t() const { return val_; }

  static const int kBits = 19;

 private:
  uint32_t val_;

  static const int kFullBits = 32;
  static_assert(kFullBits <= std::numeric_limits<uint32_t>::digits,
                "ProcessId bits layout is illegal");
  static_assert(kBits <= kFullBits, "ProcessId bits layout is illegal");
  static const int kReservedBits = kFullBits - kBits;
  static const int kNodeIndexBits = 12;
  static const int kProcessIndexBits = 7;
  static_assert(kNodeIndexBits + kProcessIndexBits == kBits, "ProcessId bits layout is illegal");
};

class DeviceId {
 public:
  DeviceId(DeviceType device_type, uint32_t device_index);
  DeviceType device_type() const;
  uint32_t device_index() const;

  // remove type cast operator in future
  operator uint32_t() const { return val_; }

  static const int kBits = 12;

 private:
  uint32_t val_;

  static const int kFullBits = 32;
  static_assert(kFullBits <= std::numeric_limits<uint32_t>::digits,
                "DeviceId bits layout is illegal");
  static_assert(kBits <= kFullBits, "DeviceId bits layout is illegal");
  static const int kReservedBits = kFullBits - kBits;
  static const int kDeviceTypeBits = 5;
  static const int kDeviceIndexBits = 7;
  static_assert(kDeviceTypeBits + kDeviceIndexBits == kBits, "DeviceId bits layout is illegal");
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

  // remove type cast operator in future
  operator uint32_t() const { return val_; }

  static const int kBits = 24;

 private:
  uint32_t val_;

  static const int kFullBits = 32;
  static_assert(kFullBits <= std::numeric_limits<uint32_t>::digits,
                "StreamId bits layout is illegal");
  static_assert(kBits <= kFullBits, "StreamId bits layout is illegal");
  static const int kReservedBits = kFullBits - kBits;
  static const int kDeviceIdBits = 12;
  static_assert(kDeviceIdBits == DeviceId::kBits, "StreamId bits layout is illegal");
  static const int kStreamIndexBits = 12;
  static_assert(kDeviceIdBits + kStreamIndexBits == kBits, "StreamId bits layout is illegal");
};

class TaskId {
 public:
  TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index);
  TaskId(uint64_t global_stream_index, uint32_t task_index);
  ProcessId process_id() const;
  StreamId stream_id() const;
  uint64_t global_stream_index() const;
  uint32_t task_index() const;
  operator uint64_t() const { return val_; }
  operator int64_t() const { return static_cast<int64_t>(val_); }
  bool operator==(const TaskId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }

 private:
  uint64_t val_;

  static const int kFullBits = 64;
  static_assert(kFullBits <= std::numeric_limits<uint64_t>::digits,
                "TaskId bits layout is illegal");
  static const int kProcessIdBits = 19;
  static_assert(kProcessIdBits == ProcessId::kBits, "TaskId bits layout is illegal");
  static const int kStreamIdBits = 24;
  static_assert(kStreamIdBits == StreamId::kBits, "TaskId bits layout is illegal");
  static const int kTaskIndexBits = 21;
  static_assert(kProcessIdBits + kStreamIdBits + kTaskIndexBits == kFullBits,
                "TaskId bits layout is illegal");
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

template<>
struct hash<oneflow::TaskType> {
  std::size_t operator()(const oneflow::TaskType& task_type) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(task_type));
  }
};

}  // namespace std

/*
namespace oneflow {

class IdUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdUtil);
  void Init();

  // StreamId & TaskId
  // common stream id
  static StreamId GetStreamId(StreamType stream_type, uint32_t device_index, uint32_t stream_index);
  // CommNet stream id
  static StreamId GetCommNetStreamId();
  // TickTock stream id
  static StreamId GetTickTockStreamId();

  // MemZoneId
  static MemZoneId GetCpuMemZoneId();
  static bool IsCpuMemZoneId(MemZoneId mem_zone_id);
  static MemZoneId GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index);
  static bool IsCudaMemZoneId(MemZoneId mem_zone_id);
  static bool IsMemZoneIdSameDevice(MemZoneId lhs, MemZoneId rhs);
  static bool IsMemZoneIdNormalUsage(MemZoneId mem_zone_id);

  // independent process task stream id
  StreamId GenerateProcessTaskIndependentStreamId(ProcessId process_id, TaskType task_type);
  // pick cpu stream id evenly
  StreamId GenerateCPUComputeStreamIdEvenly(ProcessId process_id);
  // task id
  TaskId GenerateTaskId(ProcessId process_id, StreamId stream_id);
  // chain id
  int64_t GenerateChainId(uint64_t global_stream_index);

 private:
  friend class Global<IdUtil>;
  IdUtil();
  // cfg: device_num
  // HashMap<DeviceType, uint32_t> device_type2device_num_;
  uint32_t cpu_device_num_;
  // independent generator state: map of task_type to task_num
  HashMap<std::pair<ProcessId, TaskType>, uint32_t> process_task_type2task_index_counter_;
  // task id generator state: map of process stream to task_index counter
  HashMap<uint64_t, uint32_t> process_stream2task_index_counter_;
  // cpu compute stream_id generator state: map of process_id to cpu device_index counter
  HashMap<ProcessId, uint32_t> process_id2cpu_device_index_counter_;
  // chain id generator state: map of process stream to chain_index counter
  HashMap<uint64_t, uint32_t> process_stream2chain_index_counter_;
};

}  // namespace oneflow
*/

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
