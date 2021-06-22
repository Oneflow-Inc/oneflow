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
#include "oneflow/core/memory/memory_case_registry.h"
#include "oneflow/core/common/id_util.h"

namespace oneflow {

namespace {

// MemCaseId int64_t encode
// |              | device_type | device_index  |                         |
// |              | ---- 5 ---- | ----- 7 ----- |                         |
// |              |         MemZoneId           |   pglck   | reg_by_net  |
// |              | ----------- 12 ------------ | --- 5 --- | ---- 1 ---- |
// |   reserved   |                       MemCaseId                       |
// | ---- 46 ---- | ------------------------ 18 ------------------------- |
// | ----------------------------- 64 bit ------------------------------- |

// GlobalMemCaseId int64_t encode
// |          |   rank   | MemCaseId  |
// |          | -- 19 -- | --- 18 --- |
// | reserved |    GlobalMemCaseId    |
// | -- 27 -- | -------- 37 --------- |
// | ------------ 64 bit ------------ |

constexpr size_t kRegByNetBits = 1;
constexpr size_t kPageLockedTypeBits = 5;
constexpr size_t kDeviceIndexBits = 7;
constexpr size_t kDeviceTypeBits = 5;
constexpr size_t kPageLockedTypeShift = kRegByNetBits;
constexpr size_t kDeviceIndexShift = kPageLockedTypeShift + kPageLockedTypeBits;
constexpr size_t kDeviceTypeShift = kDeviceIndexShift + kDeviceIndexBits;
constexpr size_t kRankShift = kDeviceTypeShift + kDeviceTypeBits;

}  // namespace

MemCaseId::MemCaseId(const MemoryCase& mem_case) {
  *this = MemCaseRegistryMgr<MemCaseIdGeneratorRegistry>::Get().LookupRegistry(mem_case).Generate(
      mem_case);
}

void MemCaseId::ToProto(MemoryCase* mem_case) const {
  MemCaseRegistryMgr<MemCaseIdToProtoRegistry>::Get().LookupRegistry(*this).ToProto(*this,
                                                                                    mem_case);
}

int64_t EncodeMemCaseIdToInt64(const MemCaseId& mem_case_id) {
  int64_t id = static_cast<int64_t>(mem_case_id.is_host_mem_registered_by_network());
  id |= static_cast<int64_t>(mem_case_id.host_mem_page_locked_device_type())
        << kPageLockedTypeShift;
  id |= static_cast<int64_t>(mem_case_id.device_index()) << kDeviceIndexShift;
  id |= static_cast<int64_t>(mem_case_id.device_type()) << kDeviceTypeShift;
  return id;
}

int64_t EncodeGlobalMemCaseIdToInt64(const GlobalMemCaseId& global_mem_case_id) {
  int64_t id = EncodeMemCaseIdToInt64(global_mem_case_id.mem_case_id());
  id |= static_cast<int64_t>(global_mem_case_id.node_index()) << kRankShift;
  return id;
}

bool PatchMemCase(const MemoryCase& src_mem_case, MemoryCase* dst_mem_case) {
  return MemCaseRegistryMgr<PatchMemCaseRegistry>::Get()
      .LookupRegistry(src_mem_case)
      .Patch(src_mem_case, dst_mem_case);
}

MemoryCase GenerateCorrespondingPageLockedHostMemoryCase(const MemoryCase& mem_case) {
  MemoryCase page_locked_mem_case;
  MemCaseRegistryMgr<PageLockedMemCaseRegistry>::Get().LookupRegistry(mem_case).PageLock(
      mem_case, &page_locked_mem_case);
  return page_locked_mem_case;
}

std::shared_ptr<MemoryCase> MemoryCaseUtil::MakeMemCase(const DeviceType device_type,
                                                        const int64_t device_id) {
  const auto& mem_case = std::make_shared<MemoryCase>();
  if (device_type == DeviceType::kCPU) {
    mem_case->mutable_host_mem();
  } else if (device_type == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(device_id);
  } else {
    UNIMPLEMENTED();
  }
  return mem_case;
}

}  // namespace oneflow
