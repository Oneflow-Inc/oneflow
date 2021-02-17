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
#ifndef ONEFLOW_CORE_JOB_ID_MANAGER_H_
#define ONEFLOW_CORE_JOB_ID_MANAGER_H_

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/job/task_id.pb.h"
#include <bitset>
#include <cstdint>
#include <functional>

namespace oneflow {

// ProcessId encode
// | --------- 32 bit --------- |
// | --- 20 --- | ---- 12 ----- |
// | node_index | process_index |

class ProcessId {
 public:
  static const int kBits = 32;
  static const int kLeftPartBits = 20;
  static const int kRightPartBits = 12;

  ProcessId() : val_(0) {}
  explicit ProcessId(uint32_t val) : val_(val) {}
  ProcessId(uint32_t node_index, uint32_t process_index);
  ProcessId(const ProcessId& other) { val_ = other.val_; }
  uint32_t node_index() const;
  uint32_t process_index() const;
  operator uint32_t() const { return val_; }
  operator uint64_t() const { return static_cast<uint64_t>(val_); }
  bool operator==(const ProcessId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const ProcessId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

// StreamId encode
// | ----------------------- 32 bit ------------------------ |
// | ------ 10 ------ | ------ 12 ------ | ------ 10 ------- |
// |   stream_type    |   device_index   |   stream_index    |
// | kCPUDevice       | [0, device_num)  |         0         |
// | kCudaDevice      | [0, device_num)  | enum CudaWorkType |
// | ---------------- | ---------------- | ----------------- |
// |                  | this_node_index  |  peer_node_index  |
// | kCommNet         | [0, node_num)    | [0, node_num)     |
// | ---------------- | ---------------- | ----------------- |
// | kTickTock        |        0         |         0         |
// | ---------------- | ---------------- | ----------------- |
// |                  |    task_type     |   stream_index    |
// | kIndependent     | enum TaskType    | [0, task_num)     |

enum class StreamType : int16_t {
  kInvalid = 0,
  kCPUDevice = 1,
  kCudaDevice = 2,
  kCommNet = 10,
  kTickTock = 11,
  kIndependent = 100,
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

class StreamId {
 public:
  static const int kBits = 32;
  static const int kLeftPartBits = 10;
  static const int kMiddlePartBits = 12;
  static const int kRightPartBits = 10;
  static const int kMiddleRightPartBits = kMiddlePartBits + kRightPartBits;

  StreamId() : val_(0) {}
  explicit StreamId(uint32_t val) : val_(val) {}
  StreamType stream_type() const;
  DeviceType device_type() const;
  uint32_t device_index() const;
  uint32_t stream_index() const;
  TaskType task_type() const;
  operator uint32_t() const { return val_; }
  operator uint64_t() const { return static_cast<uint64_t>(val_); }
  bool operator==(const StreamId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const StreamId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

// TaskId encode
// | ---------------- 128 bit ------------------- |
// | -- 32 -- | -- 32 --- | -- 32 -- | --- 32 --- |
// | reserved | ProcessId | StreamId | task_index |
// |                   TaskId                     |

class TaskId {
 public:
  static const int kBits = 128;
  static const int kQuarterBits = kBits / 4;
  using bits_t = std::bitset<kBits>;

  TaskId() : low_(0), high_(0) {}
  explicit TaskId(uint64_t low, uint64_t high) : low_(low), high_(high) {}
  TaskId(const TaskIdProto& task_id) : TaskId(task_id.low(), task_id.high()) {}
  TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index);
  ProcessId process_id() const { return ProcessId(static_cast<uint32_t>(high_)); }
  StreamId stream_id() const { return StreamId(static_cast<uint32_t>(low_ >> kQuarterBits)); }
  uint64_t global_stream_index() const { return (low_ >> kQuarterBits) | high_; }
  uint32_t task_index() const { return ProcessId(static_cast<uint32_t>(low_)); }
  uint64_t high() const { return high_; }
  uint64_t low() const { return low_; }
  bool operator==(const TaskId& rhs) const { return low_ == rhs.low_ && high_ == rhs.high_; }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }

 private:
  uint64_t low_;
  uint64_t high_;
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
  static const int kBits = 32;

  MemZoneId() : val_(0) {}
  explicit MemZoneId(uint32_t val) : val_(val) {}
  DeviceType device_type() const;
  uint32_t device_index() const;
  operator uint32_t() const { return val_; }
  bool operator==(const MemZoneId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

// GlobalMemZoneId encode
// | ------- 64 bit -------- |
// | --- 32 --- | --- 32 --- |
// | ProcessId  | MemZoneId  |

class GlobalMemZoneId {};

class IdUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdUtil);

  // StreamId & TaskId
  // Non-cpu device stream id
  static StreamId GetStreamId(StreamType stream_type, uint32_t device_index, uint32_t stream_index);
  static StreamId GetDeviceComputeStreamId(DeviceType device_type, uint32_t device_index);
  static StreamId GetDeviceH2DStreamId(DeviceType device_type, uint32_t device_index);
  static StreamId GetDeviceD2HStreamId(DeviceType device_type, uint32_t device_index);
  static StreamId GetDeviceMixStreamId(DeviceType device_type, uint32_t device_index);
  static StreamId GetNcclStreamId(uint32_t device_index);
  static StreamId GetCudaDecodeH2DStreamId(uint32_t device_index);
  // CPU device compute stream
  static StreamId GetCPUDeviceStreamId(uint32_t device_index);
  // CommNet
  static StreamId GetCommNetStreamId(uint32_t this_node_index, uint32_t peer_node_index);
  // TickTock
  static StreamId GetTickTockStreamId();
  // Independent
  StreamId GenerateProcessTaskIndependentStreamId(ProcessId process_id, TaskType task_type);
  // pick cpu stream id evenly
  StreamId GenerateCPUDeviceStreamIdEvenly(ProcessId process_id);
  // Task
  TaskId GenerateTaskId(ProcessId process_id, StreamId stream_id);

  // ProcessId
  static bool IsProcessIdSameNode(ProcessId lhs, ProcessId rhs);

  // MemZoneId
  static MemZoneId GetCpuMemZoneId();
  static bool IsCpuMemZoneId(MemZoneId mem_zone_id);
  static MemZoneId GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index);
  static bool IsCudaMemZoneId(MemZoneId mem_zone_id);
  static bool IsMemZoneIdSameDevice(MemZoneId lhs, MemZoneId rhs);
  static bool IsMemZoneIdNormalUsage(MemZoneId mem_zone_id);

 private:
  // cfg
  uint32_t cpu_device_num_;
  // independent generator state: map of task_type to task_num
  HashMap<std::pair<ProcessId, TaskType>, uint32_t> process_independent_task_type2task_num_;
  // task id generator state: map of process_stream to task_num
  HashMap<uint64_t, uint32_t> process_stream2task_num_;
  // cpu device stream_id generator state: map of process_id to cpu device index
  HashMap<ProcessId, uint32_t, std::hash<uint32_t>> process_id2cpu_device_index_;
};

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  // Get ThrdId, TaskId, RegstDescId
  int64_t GetGpuComputeThrdId(int64_t dev_phy_id) const { return dev_phy_id; }
  int64_t GetGpuH2DThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuD2HThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuNcclThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuMixThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuDecodeH2DThrdId(int64_t dev_phy_id) const;
  int64_t GetCpuDeviceThrdId(int64_t dev_phy_id) const;
  int64_t CommNetThrdId() const;
  int64_t TickTockThrdId() const;
  int64_t BaseIndependentThrdId() const;
  void UpdateBaseIndependentThrdId(int64_t val);

  int64_t NewTaskId(int64_t machine_id, int64_t thrd_id, int64_t local_work_stream_id);
  int64_t NewRegstDescId() { return regst_desc_id_count_++; }
  int64_t NewMemBlockId() { return mem_block_id_count_++; }
  int64_t NewChunkId() { return chunk_id_count_++; }

  // MemZoneId
  int64_t CpuMemZoneId() const { return Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum(); }
  bool IsCpuMemZone(int64_t mem_zone_id) const { return mem_zone_id == CpuMemZoneId(); }
  bool IsGpuMemZone(int64_t mem_zone_id) const { return mem_zone_id < gpu_device_num_; }
  int64_t GpuMemZoneId(int64_t dev_phy_id) const { return dev_phy_id; }
  int64_t GetGpuPhyIdFromMemZoneId(int64_t mem_zone_id) const {
    CHECK_LT(mem_zone_id, gpu_device_num_);
    return mem_zone_id;
  }

  // GetFromThrdId
  DeviceType GetDeviceTypeFromThrdId(int64_t thrd_id) const;
  int64_t GetGpuPhyIdFromThrdId(int64_t thrd_id) const;

  // Runtime
  DeviceType GetDeviceTypeFromActorId(int64_t actor_id) const;
  int64_t MachineId4ActorId(int64_t actor_id) const;
  int64_t ThrdId4ActorId(int64_t actor_id) const;

  // local_work_stream_id
  // for cpu:
  //   0: the actor thread
  // for gpu:
  //   0: the global cuda stream
  int64_t AllocateLocalWorkStreamId(int64_t machine_id, int64_t thrd_id);
  int64_t LocalWorkStreamId4TaskId(int64_t task_id) const;
  int64_t LocalWorkStreamId4ActorId(int64_t actor_id) const;
  // global_thread_id
  // sign | machine_id | thrd_id | 0  | 0
  //  1   |     10     |   11    | 21 | 21
  int64_t GlobalThrdId4TaskId(int64_t task_id) const;
  // global_work_stream_id
  // sign | machine_id | thrd_id | local_work_stream_id | 0
  //  1   |     10     |   11    |          21          | 21
  int64_t GlobalWorkStreamId4ActorId(int64_t actor_id) const;
  int64_t GlobalWorkStreamId4TaskId(int64_t task_id) const;
  int64_t AllocateChainId(int64_t global_work_stream_id);
  int64_t PickCpuThrdIdEvenly(int64_t machine_id);

 private:
  friend class Global<IDMgr>;
  IDMgr();
  int64_t GetMachineThrdId(int64_t machine_id, int64_t thrd_id);

  int64_t gpu_device_num_;
  int64_t cpu_device_num_;
  int64_t regst_desc_id_count_;
  int64_t mem_block_id_count_;
  int64_t chunk_id_count_;
  HashMap<int64_t, int64_t> machine_thrd_id2num_of_tasks_;
  HashMap<int64_t, int64_t> machine_thrd_id2stream_id_cnt_;
  HashMap<int64_t, int64_t> stream_id2chain_cnt_;
  int64_t base_independent_thrd_id_;
  HashMap<int64_t, int64_t> machine_id2num_cpu_thrd_id_picked_;

  //  64 bit id design:
  //   sign | machine | thread | local_work_stream | task
  //    1   |   10    |   11   |       21          |  21
  static const int64_t machine_id_bit_num_ = 10;
  static const int64_t thread_id_bit_num_ = 11;
  static const int64_t local_work_stream_id_bit_num_ = 21;
  static const int64_t task_id_bit_num_ = 21;
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
    auto high = std::hash<uint64_t>{}(task_id.high());
    auto low = std::hash<uint64_t>{}(task_id.low());
    return high ^ low;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
