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
#include "oneflow/core/graph/id_serialization.h"
#include <climits>

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// | node_index | process_index |                                                         |
// | --- 12 --- | ----- 7 ----- |                                                         |
// |         ProcessId          | device_type | device_index  |                           |
// | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |                           |
// |                        DeviceId                          | stream_index |            |
// | ------------------------- 31 --------------------------- | ---- 12 ---- |            |
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

namespace {

constexpr int kInt64Bits = sizeof(int64_t) * CHAR_BIT;

}  // namespace

namespace stream_id_const {

constexpr int kDeviceIndexShift = StreamId::kStreamIndexBits;
constexpr int kDeviceTypeShift = kDeviceIndexShift + DeviceId::kDeviceIndexBits;
constexpr int kProcessIndexShift = kDeviceTypeShift + DeviceId::kDeviceTypeBits;
constexpr int kNodeIndexShift = kProcessIndexShift + ProcessId::kProcessIndexBits;
static_assert(kInt64Bits == kNodeIndexShift + ProcessId::kNodeIndexBits + TaskId::kTaskIndexBits,
              "");

constexpr int64_t kStreamIndexInt64Mask = (int64_t{1} << StreamId::kStreamIndexBits) - 1;
constexpr int64_t kDeviceIndexInt64Mask = ((int64_t{1} << DeviceId::kDeviceIndexBits) - 1)
                                          << kDeviceIndexShift;
constexpr int64_t kDeviceTypeInt64Mask = ((int64_t{1} << DeviceId::kDeviceTypeBits) - 1)
                                         << kDeviceTypeShift;
constexpr int64_t kProcessIndexInt64Mask = ((int64_t{1} << ProcessId::kProcessIndexBits) - 1)
                                           << kProcessIndexShift;
constexpr int64_t kNodeIndexInt64Mask = ((int64_t{1} << ProcessId::kNodeIndexBits) - 1)
                                        << kNodeIndexShift;

}  // namespace stream_id_const

int64_t SerializeStreamIdToInt64(const StreamId& stream_id) {
  int64_t id = static_cast<int64_t>(stream_id.stream_index());
  id |= static_cast<int64_t>(stream_id.device_id().device_index())
        << stream_id_const::kDeviceIndexShift;
  id |= static_cast<int64_t>(stream_id.device_id().device_type())
        << stream_id_const::kDeviceTypeShift;
  id |= static_cast<int64_t>(stream_id.device_id().process_id().process_index())
        << stream_id_const::kProcessIndexShift;
  id |= static_cast<int64_t>(stream_id.device_id().process_id().node_index())
        << stream_id_const::kNodeIndexShift;
  return id;
}

StreamId DeserializeStreamIdFromInt64(int64_t stream_id_val) {
  int64_t node_index =
      (stream_id_val & stream_id_const::kNodeIndexInt64Mask) >> stream_id_const::kNodeIndexShift;
  int64_t process_index = (stream_id_val & stream_id_const::kProcessIndexInt64Mask)
                          >> stream_id_const::kProcessIndexShift;
  int64_t device_type =
      (stream_id_val & stream_id_const::kDeviceTypeInt64Mask) >> stream_id_const::kDeviceTypeShift;
  int64_t device_index = (stream_id_val & stream_id_const::kDeviceIndexInt64Mask)
                         >> stream_id_const::kDeviceIndexShift;
  int64_t stream_index = (stream_id_val & stream_id_const::kStreamIndexInt64Mask);

  ProcessId process_id{static_cast<node_index_t>(node_index),
                       static_cast<process_index_t>(process_index)};
  DeviceId device_id{process_id, static_cast<DeviceType>(device_type),
                     static_cast<device_index_t>(device_index)};
  return StreamId{device_id, static_cast<stream_index_t>(stream_index)};
}

namespace task_id_const {

constexpr int kStreamIndexShift = TaskId::kTaskIndexBits;
constexpr int kDeviceIndexShift = kStreamIndexShift + StreamId::kStreamIndexBits;
constexpr int kDeviceTypeShift = kDeviceIndexShift + DeviceId::kDeviceIndexBits;
constexpr int kProcessIndexShift = kDeviceTypeShift + DeviceId::kDeviceTypeBits;
constexpr int kNodeIndexShift = kProcessIndexShift + ProcessId::kProcessIndexBits;
static_assert(kInt64Bits == kNodeIndexShift + ProcessId::kNodeIndexBits, "");

constexpr int64_t kTaskIndexInt64Mask = (int64_t{1} << TaskId::kTaskIndexBits) - 1;
constexpr int64_t kStreamIndexInt64Mask = ((int64_t{1} << StreamId::kStreamIndexBits) - 1)
                                          << kStreamIndexShift;
constexpr int64_t kDeviceIndexInt64Mask = ((int64_t{1} << DeviceId::kDeviceIndexBits) - 1)
                                          << kDeviceIndexShift;
constexpr int64_t kDeviceTypeInt64Mask = ((int64_t{1} << DeviceId::kDeviceTypeBits) - 1)
                                         << kDeviceTypeShift;
constexpr int64_t kProcessIndexInt64Mask = ((int64_t{1} << ProcessId::kProcessIndexBits) - 1)
                                           << kProcessIndexShift;
constexpr int64_t kNodeIndexInt64Mask = ((int64_t{1} << ProcessId::kNodeIndexBits) - 1)
                                        << kNodeIndexShift;

}  // namespace task_id_const

int64_t SerializeTaskIdToInt64(const TaskId& task_id) {
  int64_t id = static_cast<int64_t>(task_id.task_index());
  id |= static_cast<int64_t>(task_id.stream_id().stream_index())
        << task_id_const::kStreamIndexShift;
  id |= static_cast<int64_t>(task_id.stream_id().device_id().device_index())
        << task_id_const::kDeviceIndexShift;
  id |= static_cast<int64_t>(task_id.stream_id().device_id().device_type())
        << task_id_const::kDeviceTypeShift;
  id |= static_cast<int64_t>(task_id.stream_id().device_id().process_id().process_index())
        << task_id_const::kProcessIndexShift;
  id |= static_cast<int64_t>(task_id.stream_id().device_id().process_id().node_index())
        << task_id_const::kNodeIndexShift;
  return id;
}

TaskId DeserializeTaskIdFromInt64(int64_t task_id_val) {
  int64_t node_index =
      (task_id_val & task_id_const::kNodeIndexInt64Mask) >> task_id_const::kNodeIndexShift;
  int64_t process_index =
      (task_id_val & task_id_const::kProcessIndexInt64Mask) >> task_id_const::kProcessIndexShift;
  int64_t device_type =
      (task_id_val & task_id_const::kDeviceTypeInt64Mask) >> task_id_const::kDeviceTypeShift;
  int64_t device_index =
      (task_id_val & task_id_const::kDeviceIndexInt64Mask) >> task_id_const::kDeviceIndexShift;
  int64_t stream_index =
      (task_id_val & task_id_const::kStreamIndexInt64Mask) >> task_id_const::kStreamIndexShift;
  int64_t task_index = task_id_val & task_id_const::kTaskIndexInt64Mask;
  ProcessId process_id{static_cast<node_index_t>(node_index),
                       static_cast<process_index_t>(process_index)};
  DeviceId device_id{process_id, static_cast<DeviceType>(device_type),
                     static_cast<device_index_t>(device_index)};
  StreamId stream_id{device_id, static_cast<stream_index_t>(stream_index)};
  return TaskId{stream_id, static_cast<task_index_t>(task_index)};
}

}  // namespace oneflow
