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
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

// ProcessId encode
// | -------------- 32 bit ---------------- |
// | --- 9 --- | --- 16 --- | ----- 7 ----- |
// | reserved  | node_index | process_index |

class ProcessId {
 public:
  static const int kBits = 32;
  static const int kReservedBits = 9;
  static const int kLeftBits = 16;
  static const int kRightBits = 7;
  static const int kReservedLeftBits = kReservedBits + kLeftBits;

  ProcessId() : val_(0) {}
  explicit ProcessId(uint32_t val) : val_(val) {}
  ProcessId(uint32_t node_index, uint32_t process_index);
  uint32_t node_index() const;
  uint32_t process_index() const;
  operator uint32_t() const { return val_; }
  operator uint64_t() const { return static_cast<uint64_t>(val_); }
  operator int64_t() const { return static_cast<int64_t>(val_); }
  bool operator==(const ProcessId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const ProcessId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

enum class StreamType : int16_t {
  kInvalid = 0,        // DeviceType::kInvalidDevice
  kCPU = 1,            // DeviceType::kCPU
  kCuda = 2,           // DeviceType::kGPU
  kCommNet = 20,       // DeviceType::kCPU
  kTickTock = 21,      // DeviceType::kCPU
  kIndependent = 100,  // DeviceType::kCPU
};

namespace StreamIndex {

struct CPU {
  static const uint32_t kCompute = 0;
  static const uint32_t kMax = 1;
};

struct Cuda {
  static const uint32_t kCompute = 0;
  static const uint32_t kH2D = 1;
  static const uint32_t kD2H = 2;
  static const uint32_t kMix = 3;
  static const uint32_t kNccl = 4;
  static const uint32_t kDecodeH2D = 5;
  static const uint32_t kMax = 6;
};

}  // namespace StreamIndex

// StreamId encode
// | --------------------------- 32 bit ----------------------------- |
// | -- 12 -- | ------ 8 ------ | ------ 7 ------ | ------- 5 ------- |
// | reserved |   stream_type   |   device_index  |   stream_index    |
// |          | kCPU            | [0, device_num) | StreamIndex::CPU  |
// |          | kCuda           | [0, device_num) | StreamIndex::Cuda |
// |          | --------------- | --------------- | ----------------- |
// |          | kCommNet        |        0        |         0         |
// |          | --------------- | --------------- | ----------------- |
// |          | kTickTock       |        0        |         0         |
// |          | --------------- | --------------- | ----------------- |
// |          |                 |    task_type    |   stream_index    |
// |          | kIndependent    | enum TaskType   | [0, task_num)     |

class StreamId {
 public:
  static const int kBits = 32;
  static const int kReservedBits = 12;
  static const int kLeftBits = 8;
  static const int kMiddleBits = 7;
  static const int kRightBits = 5;

  StreamId() : val_(0) {}
  explicit StreamId(uint32_t val) : val_(val) {}
  StreamType stream_type() const;
  DeviceType device_type() const;
  uint32_t device_index() const;
  uint32_t stream_index() const;
  TaskType task_type() const;
  operator uint32_t() const { return val_; }
  operator uint64_t() const { return static_cast<uint64_t>(val_); }
  operator int64_t() const { return static_cast<int64_t>(val_); }
  bool operator==(const StreamId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const StreamId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

// TaskId encode (may be extended to 128 bit in future)
// | -------------- 64 bit -------------- |
// | --- 23 --- | --- 20 --- | --- 21 --- |
// | ProcessId  |  StreamId  | task_index |
// |               TaskId                 |

class TaskId {
 public:
  static const int kBits = 64;
  static const int kLeftBits = 23;
  static const int kMiddleBits = 20;
  static const int kRightBits = 21;
  static const int kLeftMiddleBits = kLeftBits + kMiddleBits;
  static const int kMiddleRightBits = kMiddleBits + kRightBits;
  static const int kLeftRightBits = kLeftBits + kRightBits;

  TaskId() : val_(static_cast<uint64_t>(-1)) {}
  explicit TaskId(uint64_t val) : val_(val) {}
  TaskId(int64_t val) : TaskId(static_cast<uint64_t>(val)) {}
  TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index);
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
};

// MemZoneId encode
// | -------------- 32 bit --------------- |
// | ---- 12 ---- | -- 8 -- | ---- 12 ---- |
// | device_type  | usage   | device_index |

class MemZoneId {
 public:
  static const int kUsageNormal = 0;
  static const int kUsagePinnedByCuda = 1;
  static const int kUsagePinnedByNetwork = 2;
  static const int kLeftBits = 12;
  static const int kMiddleBits = 8;
  static const int kRightBits = 12;
  static const int kLeftMiddleBits = kLeftBits + kMiddleBits;
  static const int kMiddleRightBits = kMiddleBits + kRightBits;

  MemZoneId() : val_(0) {}
  explicit MemZoneId(uint32_t val) : val_(val) {}
  DeviceType device_type() const;
  uint32_t device_index() const;
  operator uint32_t() const { return val_; }
  operator int64_t() const { return static_cast<int64_t>(val_); }
  bool operator==(const MemZoneId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::ProcessId> {
  size_t operator()(const oneflow::ProcessId& process_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(process_id));
  }
};

template<>
struct hash<oneflow::StreamId> {
  size_t operator()(const oneflow::StreamId& stream_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(stream_id));
  }
};

template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(mem_zone_id));
  }
};

template<>
struct hash<oneflow::TaskId> {
  size_t operator()(const oneflow::TaskId& task_id) const {
    return std::hash<uint64_t>{}(static_cast<uint64_t>(task_id));
  }
};

}  // namespace std

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

  // Independent process task stream id
  StreamId GenerateProcessTaskIndependentStreamId(ProcessId process_id, TaskType task_type);
  // pick cpu stream id evenly
  StreamId GenerateCPUComputeStreamIdEvenly(ProcessId process_id);
  // Task
  TaskId GenerateTaskId(ProcessId process_id, StreamId stream_id);

 private:
  friend class Global<IdUtil>;
  IdUtil();
  // cfg: device_num
  // HashMap<DeviceType, uint32_t> device_type2device_num_;
  uint32_t cpu_device_num_;
  // independent generator state: map of task_type to task_num
  HashMap<std::pair<ProcessId, TaskType>, uint32_t> process_independent_task_type2task_num_;
  // task id generator state: map of process_stream to task_num
  HashMap<uint64_t, uint32_t> process_stream2task_num_;
  // cpu compute stream_id generator state: map of process_id to cpu device index
  HashMap<ProcessId, uint32_t> process_id2cpu_device_index_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
