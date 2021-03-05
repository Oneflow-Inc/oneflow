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

#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

inline bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  if (lhs.has_host_mem() && rhs.has_host_mem()) {
    const HostMemory& lhs_host_mem = lhs.host_mem();
    const HostMemory& rhs_host_mem = rhs.host_mem();
    if (lhs_host_mem.has_cuda_pinned_mem() && rhs_host_mem.has_cuda_pinned_mem()) {
      return lhs_host_mem.cuda_pinned_mem().device_id()
             == rhs_host_mem.cuda_pinned_mem().device_id();
    } else {
      return (!lhs_host_mem.has_cuda_pinned_mem()) && (!rhs_host_mem.has_cuda_pinned_mem());
    }
  }
  if (lhs.has_device_cuda_mem() && rhs.has_device_cuda_mem()) {
    return lhs.device_cuda_mem().device_id() == rhs.device_cuda_mem().device_id();
  }
  if (lhs.has_fake_dev_mem() && rhs.has_fake_dev_mem()) { return true; }
  return false;
}

struct MemoryCaseUtil {
  static bool GetCommonMemoryCase(const MemoryCase& a, const MemoryCase& b, MemoryCase* common);

  static MemoryCase GetHostPinnedMemoryCaseForRegstSeparatedHeader(const MemoryCase& mem_case);

  static int64_t GenMemZoneUniqueId(int64_t machine_id, const MemoryCase& mem_case);

  static int64_t GenMemZoneId(const MemoryCase& mem_case);

  static int64_t MergeThrdMemZoneId(int64_t thrd_id, const MemoryCase& mem_case);

  static bool IsHostUnPinnedMemoryCase(const MemoryCase& mem_case);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
