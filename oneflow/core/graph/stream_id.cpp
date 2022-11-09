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
#include "oneflow/core/graph/stream_id.h"
#include <climits>

namespace oneflow {

// StreamId encoding (bits)
// | reserved |   node_index   | device_type | device_index  | stream_index |
// | -- 21 -- | ----- 19 ----- | ---- 5 ---- | ----- 7 ----- |              |
// |          |                  DeviceId                    |              |
// |          | ------------------- 31 --------------------- | ---- 12 ---- |
// |                               StreamId                                 |
// | -------------------------------- 64 ---------------------------------- |

namespace {

constexpr size_t kInt64Bits = sizeof(int64_t) * CHAR_BIT;

constexpr size_t kDeviceIndexShift = StreamId::kStreamIndexBits;
constexpr size_t kDeviceTypeShift = kDeviceIndexShift + DeviceId::kDeviceIndexBits;
constexpr size_t kRankShift = kDeviceTypeShift + DeviceId::kDeviceTypeBits;

static_assert(kRankShift + DeviceId::kRankBits < kInt64Bits, "");

constexpr int64_t kStreamIndexInt64Mask = (int64_t{1} << StreamId::kStreamIndexBits) - 1;
constexpr int64_t kDeviceIndexInt64Mask = ((int64_t{1} << DeviceId::kDeviceIndexBits) - 1)
                                          << kDeviceIndexShift;
constexpr int64_t kDeviceTypeInt64Mask = ((int64_t{1} << DeviceId::kDeviceTypeBits) - 1)
                                         << kDeviceTypeShift;
constexpr int64_t kRankInt64Mask = ((int64_t{1} << DeviceId::kRankBits) - 1) << kRankShift;

}  // namespace

int64_t EncodeStreamIdToInt64(const StreamId& stream_id) {
  int64_t id = static_cast<int64_t>(stream_id.stream_index());
  id |= static_cast<int64_t>(stream_id.device_index()) << kDeviceIndexShift;
  id |= static_cast<int64_t>(stream_id.device_type()) << kDeviceTypeShift;
  id |= static_cast<int64_t>(stream_id.rank()) << kRankShift;
  return id;
}

StreamId DecodeStreamIdFromInt64(int64_t stream_id_val) {
  int64_t rank = (stream_id_val & kRankInt64Mask) >> kRankShift;
  int64_t device_type = (stream_id_val & kDeviceTypeInt64Mask) >> kDeviceTypeShift;
  int64_t device_index = (stream_id_val & kDeviceIndexInt64Mask) >> kDeviceIndexShift;
  int64_t stream_index = (stream_id_val & kStreamIndexInt64Mask);
  return StreamId{static_cast<DeviceId::rank_t>(rank), static_cast<DeviceType>(device_type),
                  static_cast<DeviceId::device_index_t>(device_index),
                  static_cast<StreamId::stream_index_t>(stream_index)};
}

}  // namespace oneflow
