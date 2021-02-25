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
#include "oneflow/core/common/men_zone_id_util.h"

namespace oneflow {

template<typename T>
bool CheckValueInBitsRange(T val, size_t bits) {
  static_assert(std::numeric_limits<T>::is_integer, "");
  return !static_cast<bool>(val & ~((static_cast<T>(1) << bits) - 1));
}

MemZoneId MemZoneIdUtil::GetCpuMemZoneId() {
  return MemZoneId{static_cast<uint32_t>(DeviceType::kCPU) << MemZoneId::kRightBits};
}

bool MemZoneIdUtil::IsCpuMemZoneId(const MemZoneId& mem_zone_id) {
  return ((static_cast<uint32_t>(mem_zone_id) << MemZoneId::kLeftBits) >> MemZoneId::kLeftRightBits)
         == DeviceType::kCPU;
}

MemZoneId MemZoneIdUtil::GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index) {
  CHECK(CheckValueInBitsRange(device_index, MemZoneId::kRightBits))
      << "device_index is out of range: " << device_index;
  uint32_t id = static_cast<uint32_t>(device_type) << MemZoneId::kRightBits;
  id |= device_index;
  return MemZoneId{id};
}

bool MemZoneIdUtil::IsCudaMemZoneId(const MemZoneId& mem_zone_id) {
  return ((static_cast<uint32_t>(mem_zone_id) << MemZoneId::kLeftBits)
          >> (MemZoneId::kLeftBits + MemZoneId::kRightBits))
         == DeviceType::kGPU;
}

bool MemZoneIdUtil::IsMemZoneIdSameDevice(const MemZoneId& lhs, const MemZoneId& rhs) {
  return lhs.device_type() == rhs.device_type() && lhs.device_index() == rhs.device_index();
}

int64_t MemZoneIdUtil::GetGpuPhyIdFromMemZoneId(const MemZoneId& mem_zone_id) {
  return static_cast<int64_t>((static_cast<uint32_t>(mem_zone_id) << MemZoneId::kLeftMiddleBits)
                              >> MemZoneId::kLeftMiddleBits);
}

}  // namespace oneflow
