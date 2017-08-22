#ifndef ONEFLOW_CORE_SCHEDULE_DEVICE_INFO_H_
#define ONEFLOW_CORE_SCHEDULE_DEVICE_INFO_H_

#include "oneflow/core/common/util.h"

namespace oneflow {
namespace schedule {

class DeviceInfo {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceInfo);
  DeviceInfo(const std::string dev_idx);
  ~DeviceInfo() = default;

  inline std::string DeviceIdx() const { return device_idx_; }
  inline uint32_t CpuNum() const { return cpu_num_; }
  // The ram size unit is byte.
  inline uint64_t TotalRam() const { return total_ram_; }
  inline uint64_t Ram4EachCpu() const { return ram_for_each_cpu_; }

 private:
  std::string device_idx_;
  uint32_t cpu_num_;
  uint64_t total_ram_;
  uint64_t ram_for_each_cpu_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_DEVICE_INFO_H_
