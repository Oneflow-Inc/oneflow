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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class MemCaseId {
 public:
  using device_index_t = uint32_t;

  explicit MemCaseId(const MemoryCase& mem_case);
  explicit MemCaseId(DeviceType device_type, device_index_t device_index,
                     DeviceType page_locked_device_type, bool registered_by_network)
      : device_type_(device_type),
        device_index_(device_index),
        host_mem_page_locked_device_type_(page_locked_device_type),
        host_mem_registered_by_network_(registered_by_network) {
    if (device_type != DeviceType::kCPU) {
      CHECK_EQ(page_locked_device_type, DeviceType::kInvalidDevice);
      CHECK_EQ(registered_by_network, false);
    }
  }
  explicit MemCaseId(DeviceType device_type, device_index_t device_index)
      : MemCaseId(device_type, device_index, DeviceType::kInvalidDevice, false) {}

  void ToProto(MemoryCase* mem_case) const;

  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }
  DeviceType host_mem_page_locked_device_type() const { return host_mem_page_locked_device_type_; }
  bool is_host_mem_registered_by_network() const { return host_mem_registered_by_network_; }

  bool operator==(const MemCaseId& rhs) const {
    return device_type_ == rhs.device_type_ && device_index_ == rhs.device_index_
           && host_mem_page_locked_device_type_ == rhs.host_mem_page_locked_device_type_;
  }
  bool operator!=(const MemCaseId& rhs) const { return !((*this) == rhs); }

 private:
  DeviceType device_type_;
  device_index_t device_index_;
  DeviceType host_mem_page_locked_device_type_;
  bool host_mem_registered_by_network_;
};

class GlobalMemCaseId {
 public:
  using node_index_t = uint32_t;

  explicit GlobalMemCaseId(node_index_t node_index, const MemCaseId& mem_case_id)
      : node_index_(node_index), mem_case_id_(mem_case_id) {}
  explicit GlobalMemCaseId(node_index_t node_index, const MemoryCase& mem_case)
      : GlobalMemCaseId(node_index, MemCaseId{mem_case}) {}

  node_index_t node_index() const { return node_index_; }
  const MemCaseId& mem_case_id() const { return mem_case_id_; }

  bool operator==(const GlobalMemCaseId& rhs) const {
    return node_index_ == rhs.node_index_ && mem_case_id_ == rhs.mem_case_id_;
  }
  bool operator!=(const GlobalMemCaseId& rhs) const { return !((*this) == rhs); }

 private:
  node_index_t node_index_;
  MemCaseId mem_case_id_;
};

inline bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  return MemCaseId{lhs} == MemCaseId{rhs};
}
inline bool operator!=(const MemoryCase& lhs, const MemoryCase& rhs) {
  return !(MemCaseId{lhs} == MemCaseId{rhs});
}

int64_t EncodeMemCaseIdToInt64(const MemCaseId& mem_case_id);
int64_t EncodeGlobalMemCaseIdToInt64(const GlobalMemCaseId& mem_case_id);

// Patch the source memory case to destination memory case. Patching follow below rules:
// 1) Patch failed when src_mem_case and dst_mem_case have different device_type
// or one of them has invalid device_type.
// 2) Patch failed when src_mem_case and dst_mem_case have the same non-cpu device_type
// but have different device_index.
// 3) When src_mem_case and dst_mem_case have the same cpu device_type
// and src_mem_case has more constrain than dst_mem_case(page-locked by other device,
// such as gpu or network device), patch the constrain of src_mem_case to dst_mem_case.
bool PatchMemCase(const MemoryCase& src_mem_case, MemoryCase* dst_mem_case);

// Generate host pinned memory for non-cpu device memory case
// which usually be used for generating blob header memory case for non-cpu device blob
MemoryCase GenerateCorrespondingPageLockedHostMemoryCase(const MemoryCase& mem_case);

struct MemoryCaseUtil {
  static std::shared_ptr<MemoryCase> MakeMemCase(const DeviceType device_type,
                                                 const int64_t device_id);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
