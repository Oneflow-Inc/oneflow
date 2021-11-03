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
#include "oneflow/core/graph/task_id.h"
#include <climits>

namespace oneflow {

// TaskId encoding (maybe extended to 128 bits in future)
// |            rank            | device_type | device_index  |                           |
// | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |                           |
// |                        DeviceId                          | stream_index |            |
// | ------------------------- 31 --------------------------- | ---- 12 ---- |            |
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

namespace {

constexpr size_t kInt64Bits = sizeof(int64_t) * CHAR_BIT;

constexpr size_t kStreamIndexShift = TaskId::kTaskIndexBits;
constexpr size_t kDeviceIndexShift = kStreamIndexShift + StreamId::kStreamIndexBits;
constexpr size_t kDeviceTypeShift = kDeviceIndexShift + DeviceId::kDeviceIndexBits;
constexpr size_t kRankShift = kDeviceTypeShift + DeviceId::kDeviceTypeBits;
static_assert(kInt64Bits == kRankShift + DeviceId::kRankBits, "");

constexpr int64_t kTaskIndexInt64Mask = (int64_t{1} << TaskId::kTaskIndexBits) - 1;
constexpr int64_t kStreamIndexInt64Mask = ((int64_t{1} << StreamId::kStreamIndexBits) - 1)
                                          << kStreamIndexShift;
constexpr int64_t kDeviceIndexInt64Mask = ((int64_t{1} << DeviceId::kDeviceIndexBits) - 1)
                                          << kDeviceIndexShift;
constexpr int64_t kDeviceTypeInt64Mask = ((int64_t{1} << DeviceId::kDeviceTypeBits) - 1)
                                         << kDeviceTypeShift;
constexpr int64_t kRankInt64Mask = ((int64_t{1} << DeviceId::kRankBits) - 1) << kRankShift;

}  // namespace

int64_t EncodeTaskIdToInt64(const TaskId& task_id) {
  int64_t id = static_cast<int64_t>(task_id.task_index());
  id |= static_cast<int64_t>(task_id.stream_id().stream_index()) << kStreamIndexShift;
  id |= static_cast<int64_t>(task_id.stream_id().device_index()) << kDeviceIndexShift;
  id |= static_cast<int64_t>(task_id.stream_id().device_type()) << kDeviceTypeShift;
  id |= static_cast<int64_t>(task_id.stream_id().rank()) << kRankShift;
  return id;
}

TaskId DecodeTaskIdFromInt64(int64_t task_id_val) {
  int64_t rank = (task_id_val & kRankInt64Mask) >> kRankShift;
  int64_t device_type = (task_id_val & kDeviceTypeInt64Mask) >> kDeviceTypeShift;
  int64_t device_index = (task_id_val & kDeviceIndexInt64Mask) >> kDeviceIndexShift;
  int64_t stream_index = (task_id_val & kStreamIndexInt64Mask) >> kStreamIndexShift;
  int64_t task_index = task_id_val & kTaskIndexInt64Mask;
  StreamId stream_id{static_cast<DeviceId::rank_t>(rank), static_cast<DeviceType>(device_type),
                     static_cast<DeviceId::device_index_t>(device_index),
                     static_cast<StreamId::stream_index_t>(stream_index)};
  return TaskId{stream_id, static_cast<TaskId::task_index_t>(task_index)};
}

int64_t MachineId4ActorId(int64_t actor_id) {
  return DecodeTaskIdFromInt64(actor_id).stream_id().rank();
}

int64_t ThrdId4ActorId(int64_t actor_id) {
  return EncodeStreamIdToInt64(DecodeTaskIdFromInt64(actor_id).stream_id());
}

}  // namespace oneflow
