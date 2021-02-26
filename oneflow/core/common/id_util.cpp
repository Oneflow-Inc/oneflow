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

namespace oneflow {

namespace {

template<typename T>
bool CheckValueInBitsRange(T val, size_t bits) {
  static_assert(std::numeric_limits<T>::is_integer, "");
  return !static_cast<bool>(val & ~((static_cast<T>(1) << bits) - 1));
}

}  // namespace

// ProcessId methods
ProcessId::ProcessId(uint32_t node_index, uint32_t process_index) {
  CHECK(CheckValueInBitsRange(node_index, kNodeIndexBits))
      << "node_index is out of range: " << node_index;
  CHECK(CheckValueInBitsRange(process_index, kProcessIndexBits))
      << "process_index is out of range: " << process_index;
  val_ = (node_index << kProcessIndexBits) | process_index;
}

uint32_t ProcessId::node_index() const { return val_ >> kProcessIndexBits; }

uint32_t ProcessId::process_index() const {
  return (val_ << (kFullBits - kProcessIndexBits)) >> (kFullBits - kProcessIndexBits);
}

// DeviceId methods
DeviceId::DeviceId(DeviceType device_type, uint32_t device_index) {
  CHECK(CheckValueInBitsRange(static_cast<uint32_t>(device_type), kDeviceTypeBits))
      << "device_type is out of range: " << static_cast<uint32_t>(device_type);
  CHECK(CheckValueInBitsRange(device_index, kDeviceIndexBits))
      << "device_index is out of range: " << device_index;
  val_ = (static_cast<uint32_t>(device_type) << kDeviceIndexBits) | device_index;
}

DeviceType DeviceId::device_type() const {
  return static_cast<DeviceType>(val_ >> kDeviceIndexBits);
}

uint32_t DeviceId::device_index() const {
  return (val_ << (kFullBits - kDeviceIndexBits)) >> (kFullBits - kDeviceIndexBits);
}

// StreamId methods
StreamId::StreamId(DeviceId device_id, uint32_t stream_index) {
  CHECK(CheckValueInBitsRange(device_id.val_, kDeviceIdBits))
      << "device_id is out of range: " << device_id.val_;
  CHECK(CheckValueInBitsRange(stream_index, kStreamIndexBits))
      << "stream_index is out of range: " << stream_index;
  val_ = (device_id.val_ << kStreamIndexBits) | stream_index;
}

StreamId::StreamId(DeviceType device_type, uint32_t device_index, uint32_t stream_index)
    : StreamId(DeviceId{device_type, device_index}, stream_index) {}

DeviceId StreamId::device_id() const { return DeviceId{val_ >> kStreamIndexBits}; }

DeviceType StreamId::device_type() const { return device_id().device_type(); }

uint32_t StreamId::device_index() const { return device_id().device_index(); }

uint32_t StreamId::stream_index() const {
  return (val_ << (kFullBits - kStreamIndexBits)) >> (kFullBits - kStreamIndexBits);
}

// TaskId methods
TaskId::TaskId(ProcessId process_id, StreamId stream_id, uint32_t task_index) {
  CHECK(CheckValueInBitsRange(task_index, kTaskIndexBits))
      << "task_index is out of range: " << task_index;
  CHECK(CheckValueInBitsRange(stream_id.val_, kStreamIdBits))
      << "stream_id is out of range: " << stream_id.val_;
  CHECK(CheckValueInBitsRange(process_id.val_, kProcessIdBits))
      << "process_id is out of range: " << process_id.val_;
  val_ = static_cast<uint64_t>(task_index);
  val_ |= static_cast<uint64_t>(stream_id) << kTaskIndexBits;
  val_ |= static_cast<uint64_t>(process_id) << (kTaskIndexBits + kStreamIdBits);
}

TaskId::TaskId(uint64_t global_stream_index, uint32_t task_index) {
  CHECK(CheckValueInBitsRange(global_stream_index, kProcessIdBits + kStreamIdBits))
      << "global_stream_index is out of range: " << global_stream_index;
  CHECK(CheckValueInBitsRange(task_index, kTaskIndexBits))
      << "task_index is out of range: " << task_index;
  val_ = (global_stream_index << kTaskIndexBits) | task_index;
}

ProcessId TaskId::process_id() const {
  return ProcessId{static_cast<uint32_t>(val_ >> (kTaskIndexBits + kStreamIdBits))};
}

StreamId TaskId::stream_id() const {
  return StreamId{static_cast<uint32_t>((val_ << kProcessIdBits) >> (kFullBits - kStreamIdBits))};
}

uint64_t TaskId::global_stream_index() const { return val_ >> kTaskIndexBits; }

uint32_t TaskId::task_index() const {
  return ProcessId{static_cast<uint32_t>((val_ << (kProcessIdBits + kStreamIdBits))
                                         >> (kProcessIdBits + kStreamIdBits))};
}

int64_t SerializeMemZoneIdToInt64(MemZoneId mem_zone_id) {
  return static_cast<int64_t>(mem_zone_id.val_);
}

MemZoneId DeserializeMemZoneIdFromInt64(int64_t mem_zone_id) {
  CHECK(CheckValueInBitsRange(mem_zone_id, MemZoneId::kBits))
      << "mem_zone_id is out of range: " << mem_zone_id;
  return MemZoneId{static_cast<uint32_t>(mem_zone_id)};
}

int64_t SerializeStreamIdToInt64(StreamId stream_id) {
  return static_cast<int64_t>(stream_id.val_);
}

StreamId DeserializeStreamIdFromInt64(int64_t id_val) {
  CHECK(CheckValueInBitsRange(id_val, StreamId::kBits)) << "id_val is out of range: " << id_val;
  return StreamId{static_cast<uint32_t>(id_val)};
}

int64_t SerializeTaskIdToInt64(TaskId task_id) { return static_cast<int64_t>(task_id.val_); }

TaskId DeserializeTaskIdFromInt64(int64_t id_val) {
  CHECK(CheckValueInBitsRange(id_val, TaskId::kFullBits)) << "id_val is out of range: " << id_val;
  return TaskId{static_cast<uint64_t>(id_val)};
}

}  // namespace oneflow
