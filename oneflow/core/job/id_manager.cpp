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
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

int64_t IDMgr::GetGpuH2DThrdId(int64_t dev_phy_id) const { return gpu_device_num_ + dev_phy_id; }
int64_t IDMgr::GetGpuD2HThrdId(int64_t dev_phy_id) const {
  return gpu_device_num_ * 2 + dev_phy_id;
}
int64_t IDMgr::GetGpuNcclThrdId(int64_t dev_phy_id) const {
  return gpu_device_num_ * 3 + dev_phy_id;
}
int64_t IDMgr::GetGpuMixThrdId(int64_t dev_phy_id) const {
  return gpu_device_num_ * 4 + dev_phy_id;
}
int64_t IDMgr::GetGpuDecodeH2DThrdId(int64_t dev_phy_id) const {
  return gpu_device_num_ * 5 + dev_phy_id;
}
int64_t IDMgr::GetCpuDeviceThrdId(int64_t dev_phy_id) const {
  return gpu_device_num_ * GetCudaWorkTypeSize() + dev_phy_id;
}
int64_t IDMgr::CommNetThrdId() const {
  return gpu_device_num_ * GetCudaWorkTypeSize() + cpu_device_num_;
}
int64_t IDMgr::TickTockThrdId() const { return CommNetThrdId() + 1; }
int64_t IDMgr::BaseIndependentThrdId() const { return base_independent_thrd_id_; }
void IDMgr::UpdateBaseIndependentThrdId(int64_t val) {
  if (val >= base_independent_thrd_id_) { base_independent_thrd_id_ = val + 1; }
}

int64_t IDMgr::NewTaskId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  CHECK_LT(machine_thrd_id2num_of_tasks_[machine_thrd_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  return machine_thrd_id | (machine_thrd_id2num_of_tasks_[machine_thrd_id]++);
}

DeviceType IDMgr::GetDeviceTypeFromThrdId(int64_t thrd_id) const {
  if (thrd_id < GetCudaWorkTypeSize() * gpu_device_num_) {
    return DeviceType::kGPU;
  } else {
    return DeviceType::kCPU;
  }
}

int64_t IDMgr::GetGpuPhyIdFromThrdId(int64_t thrd_id) const {
  CHECK_LT(thrd_id, GetCudaWorkTypeSize() * gpu_device_num_);
  return thrd_id % gpu_device_num_;
}

DeviceType IDMgr::GetDeviceTypeFromActorId(int64_t actor_id) const {
  int64_t thrd_id = ThrdId4ActorId(actor_id);
  return GetDeviceTypeFromThrdId(thrd_id);
}

int64_t IDMgr::MachineId4ActorId(int64_t actor_id) const {
  return actor_id >> (63 - machine_id_bit_num_);
}

int64_t IDMgr::ThrdId4ActorId(int64_t actor_id) const {
  int64_t tmp = (actor_id << machine_id_bit_num_);
  tmp &= ~(static_cast<int64_t>(1) << 63);
  return tmp >> (63 - thread_id_bit_num_);
}

int64_t IDMgr::GlobalWorkStreamId4TaskId(int64_t task_id) const {
  return (task_id >> task_id_bit_num_) << task_id_bit_num_;
}

int64_t IDMgr::GlobalWorkStreamId4ActorId(int64_t actor_id) const {
  return GlobalWorkStreamId4TaskId(actor_id);
}

int64_t IDMgr::GlobalThrdId4TaskId(int64_t task_id) const {
  int shift = local_work_stream_id_bit_num_ + task_id_bit_num_;
  return (task_id >> shift) << shift;
}

int64_t IDMgr::AllocateChainId(int64_t global_work_stream_id) {
  CHECK_LT(stream_id2chain_cnt_[global_work_stream_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  return global_work_stream_id | (stream_id2chain_cnt_[global_work_stream_id]++);
}

int64_t IDMgr::PickCpuThrdIdEvenly(int64_t machine_id) {
  return GetCpuDeviceThrdId(machine_id2num_cpu_thrd_id_picked_[machine_id]++ % cpu_device_num_);
}

IDMgr::IDMgr() {
  CHECK_LT((Global<ResourceDesc, ForSession>::Get()->TotalMachineNum()),
           static_cast<int64_t>(1) << machine_id_bit_num_);
  gpu_device_num_ = Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum();
  cpu_device_num_ = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
  CHECK_LT(gpu_device_num_ + cpu_device_num_, (static_cast<int64_t>(1) << thread_id_bit_num_) - 3);
  regst_desc_id_count_ = 0;
  mem_block_id_count_ = 0;
  chunk_id_count_ = 0;
  base_independent_thrd_id_ = TickTockThrdId() + 1;
}

int64_t IDMgr::GetMachineThrdId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_id64bit = machine_id << (63 - machine_id_bit_num_);
  int64_t thread_id64bit = thrd_id << (local_work_stream_id_bit_num_ + task_id_bit_num_);
  int64_t machine_thread_id = machine_id64bit | thread_id64bit;
  return machine_thread_id;
}

}  // namespace oneflow
