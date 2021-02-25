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

//   static MemZoneId GetCpuMemZoneId();
//   static bool IsCpuMemZoneId(MemZoneId mem_zone_id);
//   static MemZoneId GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index);
//   static bool IsCudaMemZoneId(MemZoneId mem_zone_id);
//   static bool IsMemZoneIdSameDevice(MemZoneId lhs, MemZoneId rhs);
//   static bool IsMemZoneIdNormalUsage(MemZoneId mem_zone_id);
#ifndef ONEFLOW_CORE_MEM_ZONE_ID_UTIL_H_
#define ONEFLOW_CORE_MEM_ZONE_ID_UTIL_H_

#include "oneflow/core/common/id_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/task.pb.h"
#include <limits>

namespace oneflow {

class MemZoneIdUtil {
 public:
  static MemZoneId GetCpuMemZoneId();
  static bool IsCpuMemZoneId(const MemZoneId& mem_zone_id);
  static MemZoneId GetDeviceMemZoneId(DeviceType device_type, uint32_t device_index);
  static bool IsCudaMemZoneId(const MemZoneId& mem_zone_id);
  static bool IsMemZoneIdSameDevice(const MemZoneId& lhs, const MemZoneId& rhs);
  static int64_t GetGpuPhyIdFromMemZoneId(const MemZoneId& mem_zone_id);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEM_ZONE_ID_UTIL_H_
