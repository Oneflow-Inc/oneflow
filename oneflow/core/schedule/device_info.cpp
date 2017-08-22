#include "oneflow/core/schedule/device_info.h"
#include <sys/sysinfo.h>

namespace oneflow {
namespace schedule {

DeviceInfo::DeviceInfo(const std::string dev_idx) {
  device_idx_ = dev_idx;
  struct sysinfo s_info;
  sysinfo(&s_info);
  total_ram_ = s_info.totalram * s_info.mem_unit;
  cpu_num_ = get_nprocs();
  ram_for_each_cpu_ = total_ram_ / cpu_num_;
}

}  // namespace schedule
}  // namespace oneflow
