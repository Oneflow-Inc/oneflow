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

namespace oneflow {

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

  int64_t NewTaskId(int64_t machine_id, int64_t thrd_id);
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

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
