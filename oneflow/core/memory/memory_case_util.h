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
#ifndef ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

namespace memory {

bool EqualsIgnorePinnedDevice(const MemoryCase& a, const MemoryCase& b);
void GetPinnedHostMemoryCase(const MemoryCase& mem_case, MemoryCase* ret);
MemoryCase GetPinnedHostMemoryCase(const MemoryCase& mem_case);
int64_t GetMemCaseId(const MemoryCase& mem_case);
int64_t GetUniqueMemCaseId(int64_t machine_id, const MemoryCase& mem_case);
std::shared_ptr<MemoryCase> MakeMemCaseShared(const DeviceType device_type,
                                              const int64_t device_id);
MemoryCase MakeHostMemCase();
bool IsHostMem(const MemoryCase& mem_case);

}  // namespace memory

bool operator==(const MemoryCase& lhs, const MemoryCase& rhs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
