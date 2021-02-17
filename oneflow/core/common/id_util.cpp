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
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/task_node.h"
#include <limits>

namespace oneflow {

namespace {

StreamType DeviceType2StreamType(DeviceType device_type) {
  StreamType stream_type = StreamType::kInvalid;
  switch (device_type) {
    case kCPU: stream_type = StreamType::kCPU; break;
    case kGPU: stream_type = StreamType::kCuda; break;
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
    STREAM_INDEX_CHECK_CASE(StreamType::kCPU, StreamIndex::CPU::kMax)
    STREAM_INDEX_CHECK_CASE(StreamType::kCuda, StreamIndex::Cuda::kMax)
    STREAM_INDEX_CHECK_CASE(StreamType::kCommNet, 0)
    STREAM_INDEX_CHECK_CASE(StreamType::kTickTock, 0)
    STREAM_INDEX_CHECK_CASE(StreamType::kIndependent,
                            (static_cast<uint32_t>(1) << StreamId::kRightBits))
    default: { valid = false; }
  }
  return valid;
}

template<typename T>
bool CheckValueInBitsRange(T val, size_t bits) {
  static_assert(std::numeric_limits<T>::is_integer, "");
  return !static_cast<bool>(val & ~((static_cast<T>(1) << bits) - 1));
}

}  // namespace

// ProcessId methods
ProcessId::ProcessId(uint32_t node_index, uint32_t process_index) {
  CHECK(CheckValueInBitsRange(node_index, StreamId::kLeftBits)) << "node_index is out of range";
  CHECK(CheckValueInBitsRange(process_index, StreamId::kRightBits))
      << "process_index is out of range";
  val_ = (node_index << kRightBits) | process_index;
}

uint32_t ProcessId::node_index() const { return val_ >> kRightBits; }

uint32_t ProcessId::process_index() const { return (val_ << kLeftMiddleBits) >> kLeftMiddleBits; }

// StreamId methods
StreamType StreamId::stream_type() const {
  return static_cast<StreamType>(val_ >> kMiddleRightBits);
}

DeviceType StreamId::device_type() const {
  StreamType stream_type = this->stream_type();
  DeviceType device_type = DeviceType::kInvalidDevice;
  switch (stream_type) {
    case StreamType::kCuda: {
      device_type = DeviceType::kGPU;
      break;
    }
    case StreamType::kCPU:
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
  CHECK(stream_type == StreamType::kCPU || stream_type == StreamType::kCuda)
      << "Only kCPU and kCuda stream_type support device_index()";
  return (val_ << kLeftBits) >> kLeftRightBits;
}

uint32_t StreamId::stream_index() const {
  StreamType stream_type = this->stream_type();
  CHECK(stream_type == StreamType::kCPU || stream_type == StreamType::kCuda
        || stream_type == StreamType::kIndependent)
      << "Only kCPU, kCuda, kIndependent stream_type support stream_index()";
  return (val_ << kLeftMiddleBits) >> kLeftMiddleBits;
}

TaskType StreamId::task_type() const {
  StreamType stream_type = this->stream_type();
  CHECK(stream_type == StreamType::kIndependent)
      << "Only kIndependent stream_type support task_type()";
  return static_cast<TaskType>((val_ << kLeftBits) >> kLeftRightBits);
}

// TaskId methods
TaskId::TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index) {
  CHECK(CheckValueInBitsRange(task_index, StreamId::kRightBits)) << "task_index is out of range";
  CHECK(CheckValueInBitsRange(static_cast<uint32_t>(stream_id), StreamId::kMiddleBits))
      << "stream_id is out of range";
  CHECK(CheckValueInBitsRange(static_cast<uint32_t>(process_id), StreamId::kLeftBits))
      << "process_id is out of range";
  val_ = static_cast<uint64_t>(task_index);
  val_ |= static_cast<uint64_t>(stream_id) << kRightBits;
  val_ |= static_cast<uint64_t>(process_id) << kMiddleRightBits;
}

ProcessId TaskId::process_id() const {
  uint64_t id = val_ >> kMiddleRightBits;
  return ProcessId{static_cast<uint32_t>(id)};
}

StreamId TaskId::stream_id() const {
  uint64_t id = (val_ << kLeftBits) >> kLeftRightBits;
  return StreamId{static_cast<uint32_t>(id)};
}

uint64_t TaskId::global_stream_index() const { return val_ >> kRightBits; }

uint32_t TaskId::task_index() const {
  uint64_t id = (val_ << kLeftMiddleBits) >> kLeftMiddleBits;
  return ProcessId{static_cast<uint32_t>(id)};
}

// MemZoneId methods
DeviceType MemZoneId::device_type() const {
  return static_cast<DeviceType>(val_ >> kMiddleRightBits);
}

uint32_t MemZoneId::device_index() const { return (val_ << kLeftMiddleBits) >> kLeftMiddleBits; }

// IDUtil methods
StreamId IdUtil::GetStreamId(StreamType stream_type, uint32_t device_index, uint32_t stream_index) {
  CHECK(CheckStreamIndexValid(stream_type, stream_index))
      << "invalid stream_index: " << stream_index << " under stream_type: " << stream_type;
  CHECK(CheckValueInBitsRange(device_index, StreamId::kMiddleBits))
      << "device_index is out of range: " << device_index;
  uint32_t id = 0;
  id |= static_cast<uint32_t>(stream_type) << StreamId::kMiddleRightBits;
  id |= device_index << StreamId::kRightBits;
  id |= stream_index;
  return StreamId(id);
}

StreamId IdUtil::GetCommNetStreamId(uint32_t this_node_index, uint32_t peer_node_index) {
  CHECK(CheckValueInBitsRange(this_node_index, StreamId::kMiddleBits))
      << "this_node_index is out of range";
  CHECK(CheckValueInBitsRange(peer_node_index, StreamId::kRightBits))
      << "peer_node_index is out of range";
  uint32_t id = static_cast<uint32_t>(StreamType::kCommNet) << StreamId::kMiddleRightBits;
  id |= (this_node_index << StreamId::kRightBits);
  id |= peer_node_index;
  return StreamId(id);
}

StreamId IdUtil::GetTickTockStreamId() {
  uint32_t id = static_cast<uint32_t>(StreamType::kTickTock) << StreamId::kMiddleRightBits;
  return StreamId(id);
}

MemZoneId IdUtil::GetCpuMemZoneId() {
  return MemZoneId{static_cast<uint32_t>(DeviceType::kCPU) << MemZoneId::kMiddleRightBits};
}

bool IdUtil::IsCpuMemZoneId(MemZoneId mem_zone_id) {
  return (static_cast<uint32_t>(mem_zone_id) >> MemZoneId::kMiddleRightBits) == DeviceType::kCPU;
}

MemZoneId IdUtil::GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index) {
  CHECK(CheckValueInBitsRange(device_index, StreamId::kRightBits))
      << "device_index is out of range";
  uint32_t id = static_cast<uint32_t>(device_type) << MemZoneId::kMiddleRightBits;
  id |= device_index;
  return MemZoneId{id};
}

bool IdUtil::IsCudaMemZoneId(MemZoneId mem_zone_id) {
  return (static_cast<uint32_t>(mem_zone_id) >> MemZoneId::kMiddleRightBits) == DeviceType::kGPU;
}

bool IdUtil::IsMemZoneIdSameDevice(MemZoneId lhs, MemZoneId rhs) {
  return lhs.device_type() == rhs.device_type() && lhs.device_index() == rhs.device_index();
}

bool IdUtil::IsMemZoneIdNormalUsage(MemZoneId mem_zone_id) {
  uint32_t id =
      (static_cast<uint32_t>(mem_zone_id) << MemZoneId::kLeftBits) >> MemZoneId::kMiddleRightBits;
  return id == MemZoneId::kUsageNormal;
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
  CHECK(CheckValueInBitsRange(task_num, StreamId::kRightBits));
  uint32_t id = static_cast<uint32_t>(StreamType::kIndependent) << StreamId::kMiddleRightBits;
  id |= (static_cast<uint32_t>(task_type) << StreamId::kRightBits);
  id |= static_cast<uint32_t>(task_num);
  return StreamId{id};
}

StreamId IdUtil::GenerateCPUComputeStreamIdEvenly(ProcessId process_id) {
  uint32_t device_index = process_id2cpu_device_index_[process_id]++ % cpu_device_num_;
  return GetStreamId(StreamType::kCPU, device_index, StreamIndex::CPU::kCompute);
}

TaskId IdUtil::GenerateTaskId(ProcessId process_id, StreamId stream_id) {
  uint64_t process_stream_key = static_cast<uint64_t>(process_id) << StreamId::kBits;
  process_stream_key |= static_cast<uint64_t>(stream_id);
  CHECK_LT(process_stream2task_num_[process_stream_key]++, std::numeric_limits<uint32_t>::max())
      << "task_index is out of range";
  return TaskId(process_id, stream_id, process_stream2task_num_[process_stream_key]);
}

IdUtil::IdUtil() {
  size_t machine_num = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  CHECK(CheckValueInBitsRange(machine_num, ProcessId::kLeftBits)) << "machine_num is out of range";
  cpu_device_num_ = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
  // regst_desc_id_count_ = 0;
  // mem_block_id_count_ = 0;
  // chunk_id_count_ = 0;
}

}  // namespace oneflow
