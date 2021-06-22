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
#include "oneflow/core/memory/memory_case_registry.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/to_string.h"

namespace oneflow {

namespace {

bool MatchPageableHostMemoryCase(const MemoryCase& mem_case) {
  if (!mem_case.has_host_mem()) { return false; }
  if (mem_case.host_mem().page_lock_case_case() != HostMemory::PAGE_LOCK_CASE_NOT_SET) {
    return false;
  }
  return true;
}

bool MatchPageableHostMemCaseId(const MemCaseId& mem_case_id) {
  if (mem_case_id.device_type() != DeviceType::kCPU) { return false; }
  if (mem_case_id.device_index() != 0) { return false; }
  if (mem_case_id.host_mem_page_locked_device_type() != DeviceType::kInvalidDevice) {
    return false;
  }
  return true;
}

}  // namespace

// register for normal host memory
REGISTER_MEM_CASE_ID_GENERATOR(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemoryCase)
    .SetGenerator([](const MemoryCase& mem_case) -> MemCaseId {
      CHECK(mem_case.has_host_mem());
      CHECK_EQ(mem_case.host_mem().page_lock_case_case(), HostMemory::PAGE_LOCK_CASE_NOT_SET);
      bool used_by_network =
          mem_case.host_mem().has_used_by_network() && mem_case.host_mem().used_by_network();
      return MemCaseId{DeviceType::kCPU, 0, DeviceType::kInvalidDevice, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemoryCase* mem_case) -> void {
      auto* host_mem = mem_case->mutable_host_mem();
      if (mem_case_id.is_host_mem_registered_by_network()) { host_mem->set_used_by_network(true); }
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemoryCase)
    .SetPatcher([](const MemoryCase& src_mem_case, MemoryCase* dst_mem_case) -> bool {
      CHECK(src_mem_case.has_host_mem());
      CHECK_EQ(src_mem_case.host_mem().page_lock_case_case(), HostMemory::PAGE_LOCK_CASE_NOT_SET);
      if (!dst_mem_case->has_host_mem()) { return false; }
      if (dst_mem_case->host_mem().page_lock_case_case() != HostMemory::PAGE_LOCK_CASE_NOT_SET) {
        return false;
      }
      if (src_mem_case.host_mem().has_used_by_network()
          && src_mem_case.host_mem().used_by_network()) {
        dst_mem_case->mutable_host_mem()->set_used_by_network(true);
      }
      return true;
    });

namespace {

bool MatchDummyDeviceMemoryCase(const MemoryCase& mem_case) {
  if (mem_case.has_dummy_device_mem()) { return true; }
  return false;
}

bool MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase(const MemoryCase& mem_case) {
  if (mem_case.has_host_mem() && mem_case.host_mem().has_dummy_device_pinned_mem()) {
    return true;
  } else if (mem_case.has_dummy_device_mem()) {
    return true;
  }
  return false;
}

bool MatchDummyDeviceOrDummyDevicePinnedHostMemCaseId(const MemCaseId& mem_case_id) {
  if (mem_case_id.device_type() == DeviceType::kCPU
      && mem_case_id.host_mem_page_locked_device_type() == DeviceType::kDummyDevice) {
    return true;
  } else {
    if (mem_case_id.device_type() == DeviceType::kDummyDevice) { return true; }
  }
  return false;
}

}  // namespace

// register for dummy device memory
REGISTER_MEM_CASE_ID_GENERATOR(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase)
    .SetGenerator([](const MemoryCase& mem_case) -> MemCaseId {
      DeviceType device_type = DeviceType::kInvalidDevice;
      DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
      bool used_by_network = false;
      if (mem_case.has_host_mem()) {
        CHECK(mem_case.host_mem().has_dummy_device_pinned_mem());
        device_type = DeviceType::kCPU;
        page_locked_device_type = DeviceType::kDummyDevice;
        if (mem_case.host_mem().has_used_by_network() && mem_case.host_mem().used_by_network()) {
          used_by_network = true;
        }
      } else {
        CHECK(mem_case.has_dummy_device_mem());
        device_type = DeviceType::kDummyDevice;
      }
      return MemCaseId{device_type, 0, page_locked_device_type, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemoryCase* mem_case) -> void {
      if (mem_case_id.device_type() == DeviceType::kCPU) {
        CHECK_EQ(mem_case_id.host_mem_page_locked_device_type(), DeviceType::kDummyDevice);
        auto* host_mem = mem_case->mutable_host_mem();
        host_mem->mutable_dummy_device_pinned_mem();
        if (mem_case_id.is_host_mem_registered_by_network()) {
          host_mem->set_used_by_network(true);
        }
      } else if (mem_case_id.device_type() == DeviceType::kDummyDevice) {
        mem_case->mutable_dummy_device_mem();
      } else {
        UNIMPLEMENTED();
      }
    });

REGISTER_PAGE_LOCKED_MEM_CASE(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceMemoryCase)
    .SetPageLocker([](const MemoryCase& mem_case, MemoryCase* page_locked_mem_case) -> void {
      CHECK(mem_case.has_dummy_device_mem());
      page_locked_mem_case->mutable_host_mem()->mutable_dummy_device_pinned_mem();
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase)
    .SetPatcher([](const MemoryCase& src_mem_case, MemoryCase* dst_mem_case) -> bool {
      if (src_mem_case.has_host_mem()) {
        if (!dst_mem_case->has_host_mem()) { return false; }
        if (dst_mem_case->host_mem().page_lock_case_case() == HostMemory::PAGE_LOCK_CASE_NOT_SET) {
          if (src_mem_case.host_mem().has_dummy_device_pinned_mem()) {
            dst_mem_case->mutable_host_mem()->mutable_dummy_device_pinned_mem();
          }
        } else {
          return false;
        }
        if (src_mem_case.host_mem().has_used_by_network()
            && src_mem_case.host_mem().used_by_network()) {
          dst_mem_case->mutable_host_mem()->set_used_by_network(true);
        }
      } else {
        CHECK(src_mem_case.has_dummy_device_mem());
        if (!dst_mem_case->has_dummy_device_mem()) { return false; }
      }
      return true;
    });

}  // namespace oneflow
