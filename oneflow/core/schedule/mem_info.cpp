#include "oneflow/core/schedule/mem_info.h"
#if defined(_WIN32)
// TODO
#else  // linux, osx
#include <sys/sysinfo.h>
#endif
#include <string>

namespace oneflow {
namespace schedule {

void MemInfo::GetMemInfo(const std::string this_machine_name) {
  this_machine_name_ = this_machine_name;
// get cpu memory info
#if defined(_WIN32)
// TODO
#else  // linux, osx
  struct sysinfo s_info;
  sysinfo(&s_info);
  total_cpu_ram_sz_ = s_info.totalram * s_info.mem_unit;
  cpu_num_ = get_nprocs();
#endif
  for (int32_t gpu_idx = 0; gpu_idx < GetGpuNum(); ++gpu_idx) {
    gpu_ram_sz_.push_back(GetGpuRamSize(gpu_idx));
  }
}

}  // namespace schedule
}  // namespace oneflow
