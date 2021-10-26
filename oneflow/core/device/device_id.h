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
#ifndef ONEFLOW_CORE_DEVICE_DEVICE_ID_H_
#define ONEFLOW_CORE_DEVICE_DEVICE_ID_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

// DeviceId encoding (bits)
// | reserved   |   node_index               | device_type | device_index  |
// | --- 1 ---- | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |
// |                               DeviceId                                |
// | ------------------------------- 32 ---------------------------------- |

class DeviceId {
 public:
  using index_t = uint32_t;

  constexpr static size_t kNodeIndexBits = 19;
  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static index_t kMaxNodeIndex = (index_t{1} << kNodeIndexBits) - index_t{1};
  constexpr static index_t kMaxDeviceTypeVal = (index_t{1} << kDeviceTypeBits) - index_t{1};
  constexpr static index_t kMaxDeviceIndex = (index_t{1} << kDeviceIndexBits) - index_t{1};
  constexpr static index_t kCPUDeviceIndex = 0;

  DeviceId(index_t node_index, DeviceType device_type, index_t device_index)
      : node_index_(node_index),
        device_type_(static_cast<index_t>(device_type)),
        device_index_(device_index) {
    CHECK_LE(node_index_, kMaxNodeIndex);
    CHECK_LE(device_type_, kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }

  index_t node_index() const { return node_index_; }
  DeviceType device_type() const { return static_cast<DeviceType>(device_type_); }
  index_t device_index() const { return device_index_; }

  bool operator==(const DeviceId& rhs) const {
    return node_index_ == rhs.node_index_ && device_type_ == rhs.device_type_
           && device_index_ == rhs.device_index_;
  }

  bool operator!=(const DeviceId& rhs) const { return !(*this == rhs); }

  size_t hash() const {
    size_t hash = std::hash<index_t>{}(node_index_);
    HashCombine(&hash, std::hash<index_t>{}(device_type_));
    HashCombine(&hash, std::hash<index_t>{}(device_index_));
    return hash;
  }

 private:
  index_t node_index_;
  index_t device_type_;
  index_t device_index_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DeviceId> {
  size_t operator()(const oneflow::DeviceId& device_id) const { return device_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_ID_H_
