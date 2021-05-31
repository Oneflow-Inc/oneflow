#include "oneflow/core/memory/memory_zone.h"
#include "oneflow/core/common/device_type.pb.h"
// #include <climits>

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// |            rank            | device_type | device_index  |                           |
// | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |                           |
// |                        DeviceId                          | stream_index |            |
// | ------------------------- 31 --------------------------- | ---- 12 ---- |            |
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

namespace {

// constexpr size_t kInt64Bits = sizeof(int64_t) * CHAR_BIT;

constexpr size_t kMemZoneIdDeviceTypeShift = MemZoneId::kDeviceIndexBits;

constexpr int64_t kMemZoneIdDeviceTypeInt64Mask = ((int64_t{1} << MemZoneId::kDeviceTypeBits) - 1)
                                                  << kMemZoneIdDeviceTypeShift;
constexpr int64_t kMemZoneIdDeviceIndexInt64Mask = (int64_t{1} << MemZoneId::kDeviceIndexBits) - 1;

const MemZoneId kCPUMemZoneId = MemZoneId{DeviceType::kCPU, MemZoneId::kCPUDeviceIndex};

const MemZoneId kInvalidMemZoneId = MemZoneId{DeviceType::kInvalidDevice, 0};

}  // namespace

int64_t EncodeMemZoneIdToInt64(const MemZoneId& mem_zone_id) {
  int64_t id = static_cast<int64_t>(mem_zone_id.device_index());
  id |= static_cast<int64_t>(mem_zone_id.device_type()) << kMemZoneIdDeviceTypeShift;
  return id;
}

MemZoneId DecodeMemZoneIdFromInt64(int64_t mem_zone_id) {
  int64_t device_type = (mem_zone_id & kMemZoneIdDeviceTypeInt64Mask) >> kMemZoneIdDeviceTypeShift;
  int64_t device_index = mem_zone_id & kMemZoneIdDeviceIndexInt64Mask;
  return MemZoneId(static_cast<DeviceType>(device_type),
                   static_cast<MemZoneId::device_index_t>(device_index));
}

}  // namespace oneflow
