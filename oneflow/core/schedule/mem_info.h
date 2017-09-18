#ifndef ONEFLOW_CORE_SCHEDULE_MEM_INFO_H_
#define ONEFLOW_CORE_SCHEDULE_MEM_INFO_H_

#include "oneflow/core/common/util.h"

namespace oneflow {
namespace schedule {

class MemInfo final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemInfo);
  ~MemInfo() = default;

  OF_SINGLETON(MemInfo);

  uint32_t cpu_num() const { return cpu_num_; }
  uint64_t total_cpu_ram_sz() const { return total_cpu_ram_sz_; }
  uint32_t gpu_num() const { return gpu_ram_sz_.size(); }
  uint64_t gpu_ram_sz(const uint32_t gpu_idx) const {
    return gpu_ram_sz_[gpu_idx];
  }

  const std::string& this_machine_name() const { return this_machine_name_; }
  void GetMemInfo(const std::string);

 private:
  MemInfo() = default;

  uint32_t cpu_num_;
  uint64_t total_cpu_ram_sz_;
  std::vector<uint64_t> gpu_ram_sz_;
  std::string this_machine_name_;

  uint32_t GetGpuNum() const;
  uint64_t GetGpuRamSize(int32_t) const;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_DEVICE_INFO_H_
