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

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/task.pb.h"
#include <bitset>

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

  explicit ProcessId(uint32_t val) : val_(val) {}
  ProcessId(uint32_t node_index, uint32_t process_index);
  operator uint32_t() const { return val_; }
  operator uint64_t() const { return static_cast<uint64_t>(val_); }
  uint32_t node_index() const;
  uint32_t process_index() const;

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
  kCommNet = 3,
  kTickTock = 4,
  kIndependent = 5,
};

class StreamId {
 public:
  static const int kBits = 32;
  static const int kLeftPartBits = 10;
  static const int kMiddlePartBits = 12;
  static const int kRightPartBits = 10;
  static const int kMiddleRightPartBits = kMiddlePartBits + kRightPartBits;

  explicit StreamId(uint32_t val) : val_(val) {}
  operator uint32_t() const { return val_; }
  operator uint64_t() const { return static_cast<uint64_t>(val_); }
  StreamType stream_type() const;
  DeviceType device_type() const;
  uint32_t device_index() const;
  uint32_t stream_index() const;
  TaskType task_type() const;

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
  static const int kTaskIndexBits = 32;
  using bits_t = std::bitset<kBits>;

  TaskId(uint64_t high, uint64_t low);
  TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index);
  ProcessId process_id() const;
  StreamId stream_id() const;
  uint32_t task_index() const;
  uint64_t high() const;
  uint64_t low() const;

 private:
  bits_t bits_;
};

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  // device work type stream (cuda as demonstration)
  StreamId GetDeviceComputeStreamId(DeviceType device_type, uint32_t device_index) const;
  StreamId GetDeviceH2DStreamId(DeviceType device_type, uint32_t device_index) const;
  StreamId GetDeviceD2HStreamId(DeviceType device_type, uint32_t device_index) const;
  StreamId GetDeviceMixStreamId(DeviceType device_type, uint32_t device_index) const;
  StreamId GetNcclStreamId(uint32_t device_index) const;
  StreamId GetCudaDecodeH2DStreamId(uint32_t device_index) const;

  // CPU device compute stream
  StreamId GetCPUDeviceStreamId(uint32_t device_index) const;

  // CommNet
  StreamId GetCommNetStreamId(uint32_t this_node_index, uint32_t peer_node_index) const;

  // TickTock
  StreamId GetTickTockStreamId() const;

  // Independent
  StreamId GenerateIndependentStreamId(TaskType task_type);

  // Task
  TaskId GenerateTaskId(ProcessId process_id, StreamId stream_id);

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

  // independent generator state: map of task_type to task_num
  HashMap<TaskType, size_t, std::hash<int32_t>> independent_task_type2task_num_;
  // task id generator state: map of process_stream to task_num
  HashMap<uint64_t, uint32_t> process_stream2task_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
