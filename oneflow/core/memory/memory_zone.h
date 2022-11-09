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
#ifndef ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_

#include "oneflow/core/device/device_id.h"

namespace oneflow {

using MemZoneId = DeviceId;

int64_t EncodeMemZoneIdToInt64(const MemZoneId&);
MemZoneId DecodeMemZoneIdFromInt64(int64_t);

MemZoneId GetNodeCPUMemZoneId(MemZoneId::rank_t node_index);

extern const MemZoneId kInvalidMemZoneId;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_
