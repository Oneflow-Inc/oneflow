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
/*
template<typename T> T Attr(string) 和 bool HasAttr(string)

*/
template<typename T>
T Attr(T a) {
  
}
namespace oneflow {

namespace {

bool MatchPageableHostMemoryCase(const MemoryCase& mem_case) {
  if (!mem_case.Attr("is_pinned_device").has_at_bool()) { return false; }
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
      CHECK(mem_case.Attr("device_type").at_string() == "CPU");
      bool used_by_network = mem_case.Attr("used_by_network").at_bool();
      return MemCaseId{DeviceType::kCPU, 0, DeviceType::kInvalidDevice, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemoryCase* mem_case) -> void {
      auto host_mem = mem_case->Attr("used_by_network");
      if (mem_case_id.is_host_mem_registered_by_network()) { host_mem.set_at_bool(true); }
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemoryCase)
    .SetPatcher([](const MemoryCase& src_mem_case, MemoryCase* dst_mem_case) -> bool {
      CHECK(src_mem_case.Attr("device_type").at_string() == "CPU");
      if (dst_mem_case->Attr("device_type").at_string() != "CPU") { return false; }

      if(src_mem_case.Attr("is_used_by_network").at_bool()) {
            auto dst_mem_case_host_mem = dst_mem_case->Attr("is_used_by_network");
            dst_mem_case_host_mem.set_at_bool(true);
          }
      return true;
    });

namespace {

bool MatchDummyDeviceMemoryCase(const MemoryCase& mem_case) {
  if (mem_case.Attr("device_type").at_string() != "dummy") { return false; }
  return false;
}

bool MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase(const MemoryCase& mem_case) {
  if (mem_case.Attr("host_mem").has_at_string() && mem_case.Attr("host_mem_pinned_device").at_string() == "dummy") {
    return true;
  } else if (mem_case.Attr("device_type").at_string() == "dummy") {
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
      if (mem_case.Attr("host_mem").has_at_string()) {
        CHECK(mem_case.Attr("host_mem_pinned_device").at_string() == "dummy");
        device_type = DeviceType::kCPU;
        page_locked_device_type = DeviceType::kDummyDevice;
        if (mem_case.Attr("is_used_by_network").at_bool()) {
          used_by_network = true;
        }
      } else {
        CHECK(mem_case.Attr("device_type").at_string() == "dummy");
        device_type = DeviceType::kDummyDevice;
      }
      return MemCaseId{device_type, 0, page_locked_device_type, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemoryCase* mem_case) -> void {
      if (mem_case_id.device_type() == DeviceType::kCPU) {
        CHECK_EQ(mem_case_id.host_mem_page_locked_device_type(), DeviceType::kDummyDevice);
        auto host_mem_used_by_network = mem_case->Attr("host_mem_used_by_network");
        if (mem_case_id.is_host_mem_registered_by_network()) {
          host_mem_used_by_network.set_at_bool(true);
        }
      } else if (mem_case_id.device_type() == DeviceType::kDummyDevice) {
        auto mem_case_device_type = mem_case->Attr("device_type");
        mem_case_device_type.set_at_string("dummy");
      } else {
        UNIMPLEMENTED();
      }
    });

REGISTER_PAGE_LOCKED_MEM_CASE(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceMemoryCase)
    .SetPageLocker([](const MemoryCase& mem_case, MemoryCase* page_locked_mem_case) -> void {
      CHECK(mem_case.Attr("device_type").at_string() == "dummy");
      auto dummy_device_page_locked_host_mem = page_locked_mem_case->Attr("host_mem_pinned_device");
      dummy_device_page_locked_host_mem.set_at_string("dummy");
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase)
    .SetPatcher([](const MemoryCase& src_mem_case, MemoryCase* dst_mem_case) -> bool {
      if (src_mem_case.Attr("host_mem").has_at_string()) {
        if (!dst_mem_case->Attr("host_mem").has_at_string()) { return false; }
        if (!dst_mem_case->Attr("host_mem_pinned_device").has_at_string()) {
          if (src_mem_case.Attr("host_mem_pinned_device").at_string() == "dummy") {
            auto dst_host_mem_pinned_device = dst_mem_case->Attr("host_mem_pinned_device");
            dst_host_mem_pinned_device.set_at_string("dummy");
          }
        } else {
          return false;
        }
        if (src_mem_case.Attr("host_mem_used_by_network").at_bool()) {
          auto dst_host_mem_used_by_network = dst_mem_case->Attr("host_mem_used_by_network");
          dst_host_mem_used_by_network.set_at_bool(true);
        }
      } else {
        CHECK(src_mem_case.Attr("device_type").at_string() == "dummy");
        if ( dst_mem_case->Attr("device_type").at_string() != "dummy") { return false; }
      }
      return true;
    });

}  // namespace oneflow
