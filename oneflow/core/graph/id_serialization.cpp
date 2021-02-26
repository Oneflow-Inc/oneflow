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

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// | ----------------------------------- 64 bit ----------------------------------------- |
// | ----------- 19 ----------- | ------------------- 24 ------------------- | --- 21 --- |
// |         ProcessId          |                  StreamId                  |            |
// |                            |          DeviceId           |              |            |
// | --- 12 --- | ----- 7 ----- | ---- 5 ---- | ----- 7 ----- | ---- 12 ---- | --- 21 --- |
// | node_index | process_index | device_type | device_index  | stream_index | task_index |
// |                                      TaskId                                          |

namespace {

constexpr int kNodeIndexBits = 12;
constexpr int kProcessIndexBits = 7;
constexpr int kProcessIdBits = kNodeIndexBits + kProcessIndexBits;
constexpr int kDeviceTypeBits = 5;
constexpr int kDeviceIndexBits = 7;
constexpr int kDeviceIdBits = kDeviceTypeBits + kDeviceIndexBits;
constexpr int kStreamIndexBits = 12;
constexpr int kStreamIdBits = kDeviceIdBits + kStreamIndexBits;
constexpr int kTaskIndexBits = 21;
constexpr int kFullBits = kProcessIdBits + kStreamIdBits + kTaskIndexBits;
static_assert(kFullBits == sizeof(int64_t) * CHAR_BIT, "");

}  // namespace

int64_t SerializeStreamIdToInt64(StreamId stream_id) {
  int64_t id = static_cast<int64_t>(stream_id.stream_index());
  id |= static_cast<int64_t>(stream_id.device_index()) << kStreamIndexBits;
  id |= static_cast<int64_t>(stream_id.device_type()) << (kDeviceIndexBits + kStreamIndexBits);
  return id;
}

StreamId DeserializeStreamIdFromInt64(int64_t stream_id_val) {
  int64_t device_type = stream_id_val >> (kDeviceIndexBits + kStreamIndexBits);
  int64_t device_index = (stream_id_val << kDeviceTypeBits) >> (kDeviceTypeBits + kStreamIndexBits);
  int64_t stream_index = (stream_id_val << kDeviceIdBits) >> kDeviceIdBits;
  return StreamId{static_cast<DeviceType>(device_type), static_cast<uint32_t>(device_index),
                  static_cast<uint32_t>(stream_index)};
}

int64_t SerializeTaskIdToInt64(TaskId task_id) {
  int64_t id = static_cast<int64_t>(task_id.task_index());
  id |= static_cast<int64_t>(task_id.stream_id().stream_index()) << kTaskIndexBits;
  id |= static_cast<int64_t>(task_id.stream_id().device_index())
        << (kStreamIndexBits + kTaskIndexBits);
  id |= static_cast<int64_t>(task_id.stream_id().device_type())
        << (kDeviceIndexBits + kStreamIndexBits + kTaskIndexBits);
  id |= static_cast<int64_t>(task_id.process_id().process_index())
        << (kStreamIdBits + kTaskIndexBits);
  id |= static_cast<int64_t>(task_id.process_id().node_index())
        << (kProcessIndexBits + kStreamIdBits + kTaskIndexBits);
  return id;
}

TaskId DeserializeTaskIdFromInt64(int64_t task_id_val) {
  int64_t node_index = task_id_val >> (kProcessIndexBits + kStreamIdBits + kTaskIndexBits);
  int64_t process_index = (task_id_val << kNodeIndexBits) >> (kFullBits - kProcessIndexBits);
  int64_t device_type = (task_id_val << kProcessIdBits) >> (kFullBits - kDeviceTypeBits);
  int64_t device_index =
      (task_id_val << (kProcessIdBits + kDeviceTypeBits)) >> (kFullBits - kDeviceIndexBits);
  int64_t stream_index =
      (task_id_val << (kProcessIdBits + kDeviceIdBits)) >> (kFullBits - kStreamIndexBits);
  int64_t task_index =
      (task_id_val << (kProcessIdBits + kStreamIdBits)) >> (kFullBits - kTaskIndexBits);
  ProcessId process_id{static_cast<uint32_t>(node_index), static_cast<uint32_t>(process_index)};
  StreamId stream_id{static_cast<DeviceType>(device_type), static_cast<uint32_t>(device_index),
                     static_cast<uint32_t>(stream_index)};
  return TaskId{process_id, stream_id, static_cast<uint32_t>(task_index)};
}

}  // namespace oneflow
