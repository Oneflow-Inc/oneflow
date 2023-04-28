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
#include "oneflow/core/device/device_id.h"

namespace oneflow {

namespace {
constexpr size_t kInt32Bits = sizeof(int32_t) * CHAR_BIT;

constexpr size_t kDeviceIndexShift = 0;
constexpr size_t kDeviceTypeShift = kDeviceIndexShift + DeviceId::kDeviceIndexBits;
constexpr size_t kRankShift = kDeviceTypeShift + DeviceId::kDeviceTypeBits;

static_assert(kRankShift + DeviceId::kRankBits < kInt32Bits, "");

}  // namespace

int64_t EncodeDeviceIdToInt64(const DeviceId& device_id) {
  int64_t id = static_cast<int64_t>(device_id.device_index());
  id |= static_cast<int64_t>(device_id.device_type()) << kDeviceTypeShift;
  id |= static_cast<int64_t>(device_id.rank()) << kRankShift;
  return id;
}

}  // namespace oneflow
