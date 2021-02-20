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
#include "oneflow/core/common/id_util.h"

namespace oneflow {

int64_t IDMgr::GetGpuComputeThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCuda, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::Cuda::kCompute);
}

int64_t IDMgr::GetGpuH2DThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCuda, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::Cuda::kH2D);
}

int64_t IDMgr::GetGpuD2HThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCuda, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::Cuda::kD2H);
}

int64_t IDMgr::GetGpuNcclThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCuda, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::Cuda::kNccl);
}

int64_t IDMgr::GetGpuMixThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCuda, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::Cuda::kMix);
}

int64_t IDMgr::GetGpuDecodeH2DThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCuda, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::Cuda::kDecodeH2D);
}

int64_t IDMgr::GetCpuDeviceThrdId(int64_t dev_phy_id) const {
  return IdUtil::GetStreamId(StreamType::kCPU, static_cast<uint32_t>(dev_phy_id),
                             StreamIndex::CPU::kCompute);
}

int64_t IDMgr::CommNetThrdId() const { return IdUtil::GetCommNetStreamId(); }

int64_t IDMgr::TickTockThrdId() const { return IdUtil::GetTickTockStreamId(); }

DeviceType IDMgr::GetDeviceTypeFromThrdId(int64_t thrd_id) const {
  return StreamId{static_cast<uint32_t>(thrd_id)}.device_type();
}

int64_t IDMgr::GetGpuPhyIdFromThrdId(int64_t thrd_id) const {
  StreamId stream_id{static_cast<uint32_t>(thrd_id)};
  CHECK_EQ(stream_id.device_type(), DeviceType::kGPU);
  return stream_id.device_index();
}

DeviceType IDMgr::GetDeviceTypeFromActorId(int64_t actor_id) const {
  return TaskId{static_cast<uint64_t>(actor_id)}.stream_id().device_type();
}

int64_t IDMgr::MachineId4ActorId(int64_t actor_id) const {
  return TaskId{static_cast<uint64_t>(actor_id)}.process_id().node_index();
}

int64_t IDMgr::ThrdId4ActorId(int64_t actor_id) const {
  return TaskId{static_cast<uint64_t>(actor_id)}.stream_id();
}

int64_t IDMgr::GlobalWorkStreamId4TaskId(int64_t task_id) const {
  return (task_id >> TaskId::kRightBits) << TaskId::kRightBits;
}

int64_t IDMgr::GlobalWorkStreamId4ActorId(int64_t actor_id) const {
  return GlobalWorkStreamId4TaskId(actor_id);
}

int64_t IDMgr::GlobalThrdId4TaskId(int64_t task_id) const {
  return (task_id >> TaskId::kRightBits) << TaskId::kRightBits;
}

int64_t IDMgr::AllocateChainId(int64_t global_work_stream_id) {
  return Global<IdUtil>::Get()->GenerateChainId(static_cast<uint64_t>(global_work_stream_id));
}

int64_t IDMgr::PickCpuThrdIdEvenly(int64_t machine_id) {
  return Global<IdUtil>::Get()->GenerateCPUComputeStreamIdEvenly(
      ProcessId{static_cast<uint32_t>(machine_id), 0});
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
