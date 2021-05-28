#ifndef ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

class MemZoneId {
 public:
  using device_index_t = uint32_t;

  constexpr static device_index_t kCPUDeviceIndex = 0;
  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static size_t kMaxDeviceTypeVal = (size_t{1} << kDeviceTypeBits) - size_t{1};
  constexpr static device_index_t kMaxDeviceIndex =
      (device_index_t{1} << kDeviceIndexBits) - device_index_t{1};

  MemZoneId(DeviceType device_type, device_index_t device_index)
      : device_type_(device_type), device_index_(device_index) {
    CHECK_LE(static_cast<size_t>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }

  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }

  bool operator==(const MemZoneId& rhs) const {
    return device_type_ == rhs.device_type_ && device_index_ == rhs.device_index_;
  }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }

  size_t hash() const {
    size_t hash = std::hash<size_t>{}(static_cast<size_t>(device_type_));
    HashCombine(&hash, std::hash<device_index_t>{}(device_index_));
    return hash;
  }

 private:
  DeviceType device_type_;
  device_index_t device_index_;
};

int64_t EncodeMemZoneIdToInt64(const MemZoneId&);
MemZoneId DecodeMemZoneIdFromInt64(int64_t);

extern const MemZoneId kCPUMemZoneId;

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const { return mem_zone_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ZONE_H_
