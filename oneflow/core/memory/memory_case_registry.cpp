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

namespace oneflow {

namespace {

bool MatchPageableHostMemoryCase(const MemCase& mem_case) {
  if(!mem_case.Attr<bool>("is_pinned_device")) {return false;}
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

  // namespace

// register for normal host memory
REGISTER_MEM_CASE_ID_GENERATOR(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemoryCase) 
    .SetGenerator([](const MemCase& mem_case) -> MemCaseId {
      CHECK(mem_case.Attr<DeviceType>("device_type") == kCPU);
      bool used_by_network = mem_case.Attr<bool>("used_by_network");
      return MemCaseId{DeviceType::kCPU, 0, DeviceType::kInvalidDevice, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemCase* mem_case) -> void {
      auto host_mem = mem_case->Attr<MemCase>("host_mem");
      if (mem_case_id.is_host_mem_registered_by_network()) { 
              host_mem.SetAttr<bool>("used_by_network", true); }
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kCPU)
    .SetMatcher(MatchPageableHostMemoryCase)
    .SetPatcher([](const MemCase& src_mem_case, MemCase* dst_mem_case) -> bool {
      CHECK(src_mem_case.Attr<DeviceType>("device_type") == kCPU);
      if (dst_mem_case->Attr<DeviceType>("device_type") != kCPU) { return false; }

      if(src_mem_case.Attr<bool>("is_used_by_network")) {
        dst_mem_case->SetAttr<bool>("is_used_by_network", true);
      }
      return true;
    });

namespace {

bool MatchDummyDeviceMemoryCase(const MemCase& mem_case) {
  if (mem_case.Attr<DeviceType>("device_type") != kDummyDevice) { return false; }
  return true;
}

bool MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase(const MemCase& mem_case) {
  if (mem_case.HasAttr<MemCase>("host_mem") && mem_case.Attr<DeviceType>("host_mem_pinned_device") == kDummyDevice) {
    return true;
  } else if (mem_case.Attr<DeviceType>("device_type") == kDummyDevice){
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
    .SetGenerator([](const MemCase& mem_case) -> MemCaseId {
      DeviceType device_type = DeviceType::kInvalidDevice;
      DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
      bool used_by_network = false;
      if (mem_case.Attr<DeviceType>("device_type") == kCPU) {
        CHECK(mem_case.Attr<DeviceType>("pinned_device") == kDummyDevice);
        device_type = DeviceType::kCPU;
        page_locked_device_type = DeviceType::kDummyDevice;
        if (mem_case.HasAttr<bool>("used_by_network")) {
          used_by_network = true;
        }
      } else {
        CHECK(mem_case.Attr<DeviceType>("device_type") == kDummyDevice);
        device_type = DeviceType::kDummyDevice;
      }
      return MemCaseId{device_type, 0, page_locked_device_type, used_by_network};
    });

REGISTER_MEM_CASE_ID_TO_PROTO(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemCaseId)
    .SetToProto([](const MemCaseId& mem_case_id, MemCase* mem_case) -> void {
      if (mem_case_id.device_type() == DeviceType::kCPU) {
        CHECK_EQ(mem_case_id.host_mem_page_locked_device_type(), DeviceType::kDummyDevice);
        if (mem_case_id.is_host_mem_registered_by_network()) {
          mem_case->SetAttr<bool>("host_mem_used_by_network", true);
        }
      } else if (mem_case_id.device_type() == DeviceType::kDummyDevice) {
        mem_case->SetAttr<DeviceType>("device_type", kDummyDevice);
      } else {
        UNIMPLEMENTED();
      }
    });

REGISTER_PAGE_LOCKED_MEM_CASE(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceMemoryCase)
    .SetPageLocker([](const MemCase& mem_case, MemCase* page_locked_mem_case) -> void {
      CHECK(mem_case.Attr<DeviceType>("device_type") == kDummyDevice);
      page_locked_mem_case->SetAttr<DeviceType>("host_mem_pinned_device", kDummyDevice);
    });

REGISTER_PATCH_MEM_CASE(DeviceType::kDummyDevice)
    .SetMatcher(MatchDummyDeviceOrDummyDevicePinnedHostMemoryCase)
    .SetPatcher([](const MemCase& src_mem_case, MemCase* dst_mem_case) -> bool {
      if (src_mem_case.Attr<DeviceType>("device_type") == kCPU ) {
        if (!(dst_mem_case->Attr<DeviceType>("device_type") != kCPU)) { return false; }
        if (!dst_mem_case->HasAttr<MemCase>("pinned_device")) {
          if (src_mem_case.Attr<DeviceType>("pinned_device") == kDummyDevice) {
            dst_mem_case->SetAttr<DeviceType>("pinned_device", kDummyDevice);
          }
        } else {
          return false;
        }
        if (src_mem_case.HasAttr<bool>("used_by_network")) {
          dst_mem_case->SetAttr<bool>("used_by_network", true);
        }
      } else {
        CHECK(src_mem_case.Attr<DeviceType>("device_type") == kDummyDevice);
        if ( dst_mem_case->Attr<DeviceType>("device_type") != kDummyDevice) { return false; }
      }
      return true;
    });

}  // namespace oneflow

}