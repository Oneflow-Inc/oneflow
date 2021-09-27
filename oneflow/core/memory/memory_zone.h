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

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

class MemZoneId {
 public:
  using node_index_t = DeviceId::rank_t;
  using device_index_t = DeviceId::device_index_t;

  constexpr static size_t kNodeIndexBits = DeviceId::kRankBits;
  constexpr static size_t kDeviceTypeBits = DeviceId::kDeviceTypeBits;
  constexpr static size_t kDeviceIndexBits = DeviceId::kDeviceIndexBits;

  constexpr static size_t kMaxDeviceTypeVal = DeviceId::kMaxDeviceTypeVal;
  constexpr static device_index_t kMaxDeviceIndex = DeviceId::kMaxDeviceIndex;
  constexpr static device_index_t kCPUDeviceIndex = DeviceId::kCPUDeviceIndex;

  MemZoneId() : device_id_(0, DeviceType::kInvalidDevice, 0) {}
  MemZoneId(const DeviceId& device_id) : device_id_(device_id) {}
  MemZoneId(DeviceId&& device_id) : device_id_(std::move(device_id)) {}

  MemZoneId(node_index_t node_index, DeviceType device_type, device_index_t device_index)
      : device_id_(node_index, device_type, device_index) {
    CHECK_LE(static_cast<size_t>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }

  const DeviceId& device_id() const { return device_id_; }
  node_index_t node_index() const { return device_id_.rank(); }
  DeviceType device_type() const { return device_id_.device_type(); }
  device_index_t device_index() const { return device_id_.device_index(); }

  bool operator==(const MemZoneId& rhs) const { return device_id_ == rhs.device_id_; }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }

  size_t hash() const { return device_id_.hash(); }

 private:
  DeviceId device_id_;
};

int64_t EncodeMemZoneIdToInt64(const MemZoneId&);
MemZoneId DecodeMemZoneIdFromInt64(int64_t);

MemZoneId GetNodeCPUMemZoneId(MemZoneId::node_index_t node_index);

extern const MemZoneId kInvalidMemZoneId;

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const { return mem_zone_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_
