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

#include <google/protobuf/util/message_differencer.h>

namespace oneflow {

namespace memory {

bool EqualsIgnorePinnedDevice(const MemoryCase& a, const MemoryCase& b) {
  if (a.device_type() != b.device_type()) { return false; }
  if (a.device_id() != b.device_id()) { return false; }
  return true;
}

void GetPinnedHostMemoryCase(const MemoryCase& mem_case, MemoryCase* ret) {
  ret->set_device_type(DeviceType::kCPU);
  ret->set_device_id(0);
  if (!IsHostMem(mem_case)) {
    ret->set_pinned_device_type(mem_case.device_type());
    ret->set_pinned_device_id(mem_case.device_id());
  }
}

MemoryCase GetPinnedHostMemoryCase(const MemoryCase& mem_case) {
  MemoryCase ret;
  GetPinnedHostMemoryCase(mem_case, &ret);
  return ret;
}

// clang-format off
// MemCaseId encoding (bits)
// | reserved | node_index | device_type | device_index | reserved | pinned_device_type | pinned_device_index |
// | --- 1 -- | --- 19 --- | ---- 5 ---- | ----- 7 ---- | -- 20 -- | ------- 5 -------- | ------- 7 --------- |
// | ---------------------- 32 ------------------------ | ---------------------- 32 ------------------------- |
// clang-format on

namespace {

constexpr size_t kDeviceIndexBits = 7;
constexpr size_t kDeviceTypeBits = 5;
constexpr size_t kDeviceTypeShift = kDeviceIndexBits;
constexpr size_t kNodeIndexShift = kDeviceTypeShift + kDeviceTypeBits;
constexpr size_t kPinnedDeviceShift = 32;

}  // namespace

int64_t GetMemCaseId(const MemoryCase& mem_case) {
  uint32_t high = 0;
  high |= static_cast<uint32_t>(mem_case.device_id());
  high |= static_cast<uint32_t>(mem_case.device_type()) << kDeviceTypeShift;
  uint32_t low = 0;
  if (mem_case.has_pinned_device_id()) {
    low |= static_cast<uint32_t>(mem_case.pinned_device_id());
  }
  if (mem_case.has_pinned_device_type()) {
    low |= static_cast<uint32_t>(mem_case.pinned_device_type()) << kDeviceTypeShift;
  }
  int64_t id = 0;
  id |= static_cast<int64_t>(high) << kPinnedDeviceShift;
  id |= static_cast<int64_t>(low);
  return id;
}

int64_t GetUniqueMemCaseId(int64_t machine_id, const MemoryCase& mem_case) {
  int64_t id = 0;
  id |= (machine_id << kNodeIndexShift << kPinnedDeviceShift);
  id |= GetMemCaseId(mem_case);
  return id;
}

std::shared_ptr<MemoryCase> MakeMemCaseShared(const DeviceType device_type,
                                              const int64_t device_id) {
  auto mem_case_ptr = std::make_shared<MemoryCase>();
  mem_case_ptr->set_device_type(device_type);
  // We consider that there is only one cpu physical device.
  // As non-cpu devices, a logical device map to a physical device,
  // however as cpu devices, all logical devices map to a single physical device.
  if (device_type == DeviceType::kCPU) {
    mem_case_ptr->set_device_id(0);
  } else {
    mem_case_ptr->set_device_id(device_id);
  }
  return mem_case_ptr;
}

MemoryCase MakeHostMemCase() {
  MemoryCase mem_case;
  mem_case.set_device_type(DeviceType::kCPU);
  mem_case.set_device_id(0);
  return mem_case;
}

bool IsHostMem(const MemoryCase& mem_case) { return mem_case.device_type() == DeviceType::kCPU; }

}  // namespace memory

bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  return google::protobuf::util::MessageDifferencer::Equals(lhs, rhs);
}

}  // namespace oneflow
