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
#ifndef ONEFLOW_CORE_COMMON_ID_UTIL_H_
#define ONEFLOW_CORE_COMMON_ID_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/task.pb.h"
#include <limits>

namespace oneflow {

// MemZoneId encode
// | ---------------- 32 bit ----------------- |
// | ---- 20 ---- | ---- 5 ---- | ---- 7 ----  |
// |   reserved   | device_type | device_index |

class MemZoneId {
 public:
  static const int kBits = 32;
  static const int kLeftBits = 20;
  static const int kMiddleBits = 5;
  static const int kRightBits = 7;
  static const int kLeftMiddleBits = kLeftBits + kMiddleBits;
  static const int kMiddleRightBits = kMiddleBits + kRightBits;
  static const int kLeftRightBits = kLeftBits + kRightBits;
  static_assert(kBits <= std::numeric_limits<uint32_t>::digits, "MemZoneId bits layout is illegal");
  static_assert(kBits == kLeftBits + kMiddleBits + kRightBits, "MemZoneId bits layout is illegal");

  MemZoneId() : val_(0) {}
  explicit MemZoneId(uint32_t val) : val_(val) {}
  DeviceType device_type() const;
  uint32_t device_index() const;
  operator uint32_t() const { return val_; }
  operator int64_t() const { return static_cast<int64_t>(val_); }
  bool operator==(const MemZoneId& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }

 private:
  uint32_t val_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(mem_zone_id));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
