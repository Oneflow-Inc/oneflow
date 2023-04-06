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
  using rank_t = uint32_t;
  using device_type_t = uint32_t;
  using device_index_t = uint32_t;

  constexpr static size_t kRankBits = 16;
  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static rank_t kMaxRank = (rank_t{1} << kRankBits) - rank_t{1};
  constexpr static device_type_t kMaxDeviceTypeVal =
      (device_type_t{1} << kDeviceTypeBits) - device_type_t{1};
  constexpr static device_index_t kMaxDeviceIndex =
      (device_index_t{1} << kDeviceIndexBits) - device_index_t{1};

  DeviceId(rank_t rank, DeviceType device_type, device_index_t device_index)
      : rank_(rank),
        device_type_(static_cast<device_type_t>(device_type)),
        device_index_(device_index) {
    CHECK_LE(rank_, kMaxRank);
    CHECK_LE(device_type_, kMaxDeviceTypeVal);
    CHECK_LE(device_index_, kMaxDeviceIndex);
  }

  rank_t rank() const { return rank_; }
  DeviceType device_type() const { return static_cast<DeviceType>(device_type_); }
  device_index_t device_index() const { return device_index_; }

  bool operator==(const DeviceId& rhs) const {
    return rank_ == rhs.rank_ && device_type_ == rhs.device_type_
           && device_index_ == rhs.device_index_;
  }

  bool operator!=(const DeviceId& rhs) const { return !(*this == rhs); }

  size_t hash() const {
    size_t hash = std::hash<rank_t>{}(rank_);
    HashCombine(&hash, std::hash<device_type_t>{}(device_type_));
    HashCombine(&hash, std::hash<device_index_t>{}(device_index_));
    return hash;
  }

 private:
  rank_t rank_;
  device_type_t device_type_;
  device_index_t device_index_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DeviceId> {
  size_t operator()(const oneflow::DeviceId& device_id) const { return device_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_ID_H_
