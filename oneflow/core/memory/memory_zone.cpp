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
#include "oneflow/core/memory/memory_zone.h"
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

namespace {

// constexpr size_t kInt64Bits = sizeof(int64_t) * CHAR_BIT;

constexpr size_t kMemZoneIdDeviceTypeShift = MemZoneId::kDeviceIndexBits;

constexpr int64_t kMemZoneIdDeviceTypeInt64Mask = ((int64_t{1} << MemZoneId::kDeviceTypeBits) - 1)
                                                  << kMemZoneIdDeviceTypeShift;
constexpr int64_t kMemZoneIdDeviceIndexInt64Mask = (int64_t{1} << MemZoneId::kDeviceIndexBits) - 1;

}  // namespace

const MemZoneId kCPUMemZoneId = MemZoneId{DeviceType::kCPU, MemZoneId::kCPUDeviceIndex};

const MemZoneId kInvalidMemZoneId = MemZoneId{DeviceType::kInvalidDevice, 0};

int64_t EncodeMemZoneIdToInt64(const MemZoneId& mem_zone_id) {
  int64_t id = static_cast<int64_t>(mem_zone_id.device_index());
  id |= static_cast<int64_t>(mem_zone_id.device_type()) << kMemZoneIdDeviceTypeShift;
  return id;
}

MemZoneId DecodeMemZoneIdFromInt64(int64_t mem_zone_id) {
  int64_t device_type = (mem_zone_id & kMemZoneIdDeviceTypeInt64Mask) >> kMemZoneIdDeviceTypeShift;
  int64_t device_index = mem_zone_id & kMemZoneIdDeviceIndexInt64Mask;
  return MemZoneId(static_cast<DeviceType>(device_type),
                   static_cast<MemZoneId::device_index_t>(device_index));
}

}  // namespace oneflow
