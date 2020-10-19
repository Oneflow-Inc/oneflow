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
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

bool MemoryCaseUtil::GetCommonMemoryCase(const MemoryCase& a, const MemoryCase& b,
                                         MemoryCase* common) {
  if (a.has_device_cuda_mem() && b.has_device_cuda_mem()) {
    if (a.device_cuda_mem().device_id() == b.device_cuda_mem().device_id()) {
      *common = a;
      return true;
    } else {
      return false;
    }
  } else if (a.has_host_mem() && b.has_host_mem()) {
    *common = a;
    if (b.host_mem().has_cuda_pinned_mem()) {
      *common->mutable_host_mem()->mutable_cuda_pinned_mem() = b.host_mem().cuda_pinned_mem();
    }
    if (b.host_mem().has_used_by_network()) {
      common->mutable_host_mem()->set_used_by_network(true);
    }
    return true;
  } else {
    return false;
  }
}

MemoryCase MemoryCaseUtil::GetHostPinnedMemoryCaseForRegstSeparatedHeader(
    const MemoryCase& mem_case) {
  CHECK(mem_case.has_device_cuda_mem());
  MemoryCase ret;
  ret.mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(
      mem_case.device_cuda_mem().device_id());
  return ret;
}

int64_t MemoryCaseUtil::GenMemZoneUniqueId(int64_t machine_id, const MemoryCase& mem_case) {
  int64_t mem_zone_id = 1024;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      mem_zone_id = 1025 + mem_case.host_mem().cuda_pinned_mem().device_id();
    }
  } else {
    mem_zone_id = mem_case.device_cuda_mem().device_id();
  }
  return (machine_id << 32) | mem_zone_id;
}

bool MemoryCaseUtil::IsHostUnPinnedMemoryCase(const MemoryCase& mem_case) {
  return mem_case.has_host_mem() && !mem_case.host_mem().has_cuda_pinned_mem()
         && !mem_case.host_mem().used_by_network();
}

}  // namespace oneflow
