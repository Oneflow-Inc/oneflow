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
#include "oneflow/core/graph/task_node.h"
#include <limits>

namespace oneflow {

namespace {

StreamType DeviceType2StreamType(DeviceType device_type) {
  StreamType stream_type = StreamType::kInvalid;
  switch (device_type) {
    case kCPU: stream_type = StreamType::kCPUDevice; break;
    case kGPU: stream_type = StreamType::kCudaDevice; break;
    default: UNIMPLEMENTED();
  }
  return stream_type;
}

bool CheckStreamIndexValid(StreamType stream_type, uint32_t stream_index) {
  bool valid = false;
#define STREAM_INDEX_CHECK_CASE(case_type, index_max) \
  case case_type: {                                   \
    if (stream_index < index_max) { valid = true; }   \
    break;                                            \
  }

  switch (stream_type) {
    STREAM_INDEX_CHECK_CASE(StreamType::kCPUDevice, StreamIndex::CPU::kMax)
    STREAM_INDEX_CHECK_CASE(StreamType::kCudaDevice, StreamIndex::Cuda::kMax)
    STREAM_INDEX_CHECK_CASE(StreamType::kCommNet, 0)
    STREAM_INDEX_CHECK_CASE(StreamType::kTickTock, 0)
    STREAM_INDEX_CHECK_CASE(StreamType::kIndependent,
                            (static_cast<uint32_t>(1) << StreamId::kRightPartBits))
    default: { valid = false; }
  }
  return valid;
}

template<typename T>
bool CheckValueInBitsRange(T val, int bits) {
  static_assert(std::numeric_limits<T>::is_integer, "");
  return !static_cast<bool>(val & ~((static_cast<T>(1) << bits) - 1));
}

}  // namespace

// ProcessId methods
ProcessId::ProcessId(uint32_t node_index, uint32_t process_index) {
  CHECK(CheckValueInBitsRange(node_index, StreamId::kLeftPartBits)) << "node_index is out of range";
  CHECK(CheckValueInBitsRange(process_index, StreamId::kRightPartBits))
      << "process_index is out of range";
  val_ = (node_index << kRightPartBits) | process_index;
}

uint32_t ProcessId::node_index() const { return val_ >> kRightPartBits; }

uint32_t ProcessId::process_index() const { return (val_ << kLeftPartBits) >> kLeftPartBits; }

// StreamId methods
StreamType StreamId::stream_type() const {
  return static_cast<StreamType>(val_ >> (kMiddlePartBits + kRightPartBits));
}

DeviceType StreamId::device_type() const {
  StreamType stream_type = this->stream_type();
  DeviceType device_type = DeviceType::kInvalidDevice;
  switch (stream_type) {
    case StreamType::kCudaDevice: {
      device_type = DeviceType::kGPU;
      break;
    }
    case StreamType::kCPUDevice:
    case StreamType::kCommNet:
    case StreamType::kTickTock:
    case StreamType::kIndependent: {
      device_type = DeviceType::kCPU;
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  return device_type;
}

uint32_t StreamId::device_index() const {
  StreamType stream_type = this->stream_type();
  CHECK(stream_type == StreamType::kCPUDevice || stream_type == StreamType::kCudaDevice)
      << "Only kCPUDevice and kCudaDevice stream_type support device_index()";
  return (val_ << kLeftPartBits) >> (kLeftPartBits + kRightPartBits);
}

uint32_t StreamId::stream_index() const {
  StreamType stream_type = this->stream_type();
  CHECK(stream_type == StreamType::kCPUDevice || stream_type == StreamType::kCudaDevice
        || stream_type == StreamType::kIndependent)
      << "Only kCPUDevice, kCudaDevice, kIndependent stream_type support stream_index()";
  return (val_ << (kLeftPartBits + kMiddlePartBits)) >> (kLeftPartBits + kMiddlePartBits);
}

TaskType StreamId::task_type() const {
  StreamType stream_type = this->stream_type();
  CHECK(stream_type == StreamType::kIndependent)
      << "Only kIndependent stream_type support task_type()";
  return static_cast<TaskType>((val_ << kLeftPartBits) >> (kLeftPartBits + kRightPartBits));
}

// TaskId methods
TaskId::TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index) {
  low_ = 0;
  low_ |= static_cast<uint64_t>(stream_id) << kQuarterBits;
  low_ |= static_cast<uint64_t>(task_index);
  high_ = 0;
  high_ |= static_cast<uint64_t>(process_id);
}

void TaskId::to_proto(TaskIdProto* proto) const {
  proto->set_low(low_);
  proto->set_high(high_);
}

ProcessId TaskId::process_id() const { return ProcessId(static_cast<uint32_t>(high_)); }

StreamId TaskId::stream_id() const { return StreamId(static_cast<uint32_t>(low_ >> kQuarterBits)); }

uint64_t TaskId::global_stream_index() const {
  return (low_ >> kQuarterBits) | (high_ << kQuarterBits);
}

uint32_t TaskId::task_index() const {
  uint64_t id = (low_ << kQuarterBits) >> kQuarterBits;
  return ProcessId(static_cast<uint32_t>(id));
}

// MemZoneId methods
DeviceType MemZoneId::device_type() const {
  return static_cast<DeviceType>(val_ >> kMiddleRightPartBits);
}

uint32_t MemZoneId::device_index() const {
  return (val_ << kLeftMiddlePartBits) >> kLeftMiddlePartBits;
}

// IDUtil methods
StreamId IdUtil::GetStreamId(StreamType stream_type, uint32_t device_index, uint32_t stream_index) {
  CHECK(CheckStreamIndexValid(stream_type, stream_index))
      << "invalid stream_index: " << stream_index << " under stream_type: " << stream_type;
  CHECK(CheckValueInBitsRange(device_index, StreamId::kMiddlePartBits))
      << "device_index is out of range: " << device_index;
  uint32_t id = 0;
  id |= static_cast<uint32_t>(stream_type) << StreamId::kMiddleRightPartBits;
  id |= device_index << StreamId::kRightPartBits;
  id |= stream_index;
  return StreamId(id);
}

StreamId IdUtil::GetDeviceComputeStreamId(DeviceType device_type, uint32_t device_index) {
  StreamType stream_type = DeviceType2StreamType(device_type);
  uint32_t id = (static_cast<uint32_t>(stream_type) << StreamId::kMiddleRightPartBits)
                | (device_index << StreamId::kRightPartBits);
  return StreamId(id);
}

StreamId IdUtil::GetDeviceH2DStreamId(DeviceType device_type, uint32_t device_index) {
  StreamType stream_type = DeviceType2StreamType(device_type);
  uint32_t id = ((static_cast<uint32_t>(stream_type) << StreamId::kMiddleRightPartBits)
                 | (device_index << StreamId::kRightPartBits))
                + 1;
  return StreamId(id);
}

StreamId IdUtil::GetDeviceD2HStreamId(DeviceType device_type, uint32_t device_index) {
  StreamType stream_type = DeviceType2StreamType(device_type);
  uint32_t id = ((static_cast<uint32_t>(stream_type) << StreamId::kMiddleRightPartBits)
                 | (device_index << StreamId::kRightPartBits))
                + 2;
  return StreamId(id);
}

StreamId IdUtil::GetDeviceMixStreamId(DeviceType device_type, uint32_t device_index) {
  StreamType stream_type = DeviceType2StreamType(device_type);
  uint32_t id = ((static_cast<uint32_t>(stream_type) << StreamId::kMiddleRightPartBits)
                 | (device_index << StreamId::kRightPartBits))
                + 3;
  return StreamId(id);
}

StreamId IdUtil::GetNcclStreamId(uint32_t device_index) {
  uint32_t id = ((static_cast<uint32_t>(StreamType::kCudaDevice) << StreamId::kMiddleRightPartBits)
                 | (device_index << StreamId::kRightPartBits))
                + 4;
  return StreamId(id);
}

StreamId IdUtil::GetCudaDecodeH2DStreamId(uint32_t device_index) {
  int32_t id = ((static_cast<int32_t>(StreamType::kCudaDevice) << StreamId::kMiddleRightPartBits)
                | (device_index << StreamId::kRightPartBits))
               + 5;
  return StreamId(id);
}

StreamId IdUtil::GetCPUDeviceStreamId(uint32_t device_index) {
  uint32_t id = (static_cast<uint32_t>(StreamType::kCPUDevice) << StreamId::kMiddleRightPartBits)
                | (device_index << StreamId::kRightPartBits);
  return StreamId(id);
}

StreamId IdUtil::GetCommNetStreamId(uint32_t this_node_index, uint32_t peer_node_index) {
  CHECK(CheckValueInBitsRange(this_node_index, StreamId::kMiddlePartBits))
      << "this_node_index is out of range";
  CHECK(CheckValueInBitsRange(peer_node_index, StreamId::kRightPartBits))
      << "peer_node_index is out of range";
  uint32_t id = static_cast<uint32_t>(StreamType::kCommNet) << StreamId::kMiddleRightPartBits;
  id |= (this_node_index << StreamId::kRightPartBits);
  id |= peer_node_index;
  return StreamId(id);
}

StreamId IdUtil::GetTickTockStreamId() {
  uint32_t id = static_cast<uint32_t>(StreamType::kTickTock) << StreamId::kMiddleRightPartBits;
  return StreamId(id);
}

StreamId IdUtil::GenerateProcessTaskIndependentStreamId(ProcessId process_id, TaskType task_type) {
  auto key = std::make_pair(process_id, task_type);
  if (process_independent_task_type2task_num_.find(key)
      == process_independent_task_type2task_num_.end()) {
    process_independent_task_type2task_num_[key] = 0;
  }
  uint32_t& task_num = process_independent_task_type2task_num_[key];
  task_num += 1;
  if (IsClassRegistered<int32_t, IndependentThreadNum4TaskType>(task_type)) {
    std::unique_ptr<IndependentThreadNum4TaskType> idp_thrd_num_ptr(
        NewObj<int32_t, IndependentThreadNum4TaskType>(task_type));
    if (task_num > *idp_thrd_num_ptr) { task_num %= *idp_thrd_num_ptr; }
  }
  CHECK(CheckValueInBitsRange(task_num, StreamId::kRightPartBits));
  uint32_t id = static_cast<uint32_t>(StreamType::kIndependent) << StreamId::kMiddleRightPartBits;
  id |= (static_cast<uint32_t>(task_type) << StreamId::kRightPartBits);
  id |= static_cast<uint32_t>(task_num);
  return StreamId(id);
}

StreamId IdUtil::GenerateCPUDeviceStreamIdEvenly(ProcessId process_id) {
  uint32_t device_index = process_id2cpu_device_index_[process_id]++ % cpu_device_num_;
  return GetCPUDeviceStreamId(device_index);
}

TaskId IdUtil::GenerateTaskId(ProcessId process_id, StreamId stream_id) {
  uint64_t process_stream_key =
      (static_cast<uint64_t>(process_id) << ProcessId::kBits) | static_cast<uint64_t>(stream_id);
  if (process_stream2task_num_.find(process_stream_key) == process_stream2task_num_.end()) {
    process_stream2task_num_[process_stream_key] = 0;
  }
  CHECK_LT(process_stream2task_num_[process_stream_key], std::numeric_limits<uint32_t>::max());
  process_stream2task_num_[process_stream_key] += 1;
  return TaskId(process_id, stream_id, process_stream2task_num_[process_stream_key]);
}

MemZoneId IdUtil::GetCpuMemZoneId() {
  return MemZoneId{static_cast<uint32_t>(DeviceType::kCPU) << MemZoneId::kMiddleRightPartBits};
}

bool IdUtil::IsCpuMemZoneId(MemZoneId mem_zone_id) {
  return (static_cast<uint32_t>(mem_zone_id) >> MemZoneId::kMiddleRightPartBits)
         == DeviceType::kCPU;
}

MemZoneId IdUtil::GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index) {
  CHECK(CheckValueInBitsRange(device_index, StreamId::kRightPartBits))
      << "device_index is out of range";
  uint32_t id = static_cast<uint32_t>(device_type) << MemZoneId::kMiddleRightPartBits;
  id |= device_index;
  return MemZoneId{id};
}

bool IdUtil::IsCudaMemZoneId(MemZoneId mem_zone_id) {
  return (static_cast<uint32_t>(mem_zone_id) >> MemZoneId::kMiddleRightPartBits)
         == DeviceType::kGPU;
}

bool IdUtil::IsMemZoneIdSameDevice(MemZoneId lhs, MemZoneId rhs) {
  return lhs.device_type() == rhs.device_type() && lhs.device_index() == rhs.device_index();
}

bool IdUtil::IsMemZoneIdNormalUsage(MemZoneId mem_zone_id) {
  uint32_t id = (static_cast<uint32_t>(mem_zone_id) << MemZoneId::kLeftPartBits)
                >> MemZoneId::kMiddleRightPartBits;
  return id == MemZoneId::kUsageNormal;
}

IdUtil::IdUtil() {
  size_t machine_num = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  CHECK(CheckValueInBitsRange(machine_num, ProcessId::kLeftPartBits));
  cpu_device_num_ = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
  // regst_desc_id_count_ = 0;
  // mem_block_id_count_ = 0;
  // chunk_id_count_ = 0;
}

// old methods (deprecated)
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

int64_t IDMgr::NewTaskId(int64_t machine_id, int64_t thrd_id, int64_t local_work_stream_id) {
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  CHECK_LT(machine_thrd_id2num_of_tasks_[machine_thrd_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  CHECK_LT(local_work_stream_id, static_cast<int64_t>(1) << local_work_stream_id_bit_num_);
  return machine_thrd_id | (local_work_stream_id << task_id_bit_num_)
         | (machine_thrd_id2num_of_tasks_[machine_thrd_id]++);
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

int64_t IDMgr::AllocateLocalWorkStreamId(int64_t machine_id, int64_t thrd_id) {
  return 100 + (machine_thrd_id2stream_id_cnt_[GetMachineThrdId(machine_id, thrd_id)]++);
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

int64_t IDMgr::LocalWorkStreamId4TaskId(int64_t task_id) const {
  int64_t tmp = (task_id << (machine_id_bit_num_ + thread_id_bit_num_));
  tmp &= ~(static_cast<int64_t>(1) << 63);
  return tmp >> (63 - local_work_stream_id_bit_num_);
}

int64_t IDMgr::LocalWorkStreamId4ActorId(int64_t actor_id) const {
  return LocalWorkStreamId4TaskId(actor_id);
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
}

int64_t IDMgr::GetMachineThrdId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_id64bit = machine_id << (63 - machine_id_bit_num_);
  int64_t thread_id64bit = thrd_id << (local_work_stream_id_bit_num_ + task_id_bit_num_);
  int64_t machine_thread_id = machine_id64bit | thread_id64bit;
  return machine_thread_id;
}

}  // namespace oneflow
