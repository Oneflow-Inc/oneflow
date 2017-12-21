#include "oneflow/core/memory/memory_device.h"
#include "oneflow/core/job/id_manager.h"

#if defined(_WIN32)
// TODO
#else
#include <sys/sysinfo.h>
#endif

namespace oneflow {

size_t MemoryDeviceMgr::GetThisMachineHostMemSize() const {
#if defined(_WIN32)
  //  TODO
  UNEXPECTED_RUN();
#else  // linux, osx
  struct sysinfo s_info;
  sysinfo(&s_info);
  return s_info.totalram * s_info.mem_unit;
#endif
}

MemoryDeviceMgr::MemoryDeviceMgr() {
  host_mem_size_ = GetThisMachineHostMemSize();
  dev_mem_size_ = GetThisMachineDeviceMemSize();
}

MemoryDevice::MemoryDevice(uint32_t machine_id, const MemoryCase& mem_case)
    : machine_id_(machine_id),
      memory_type_(mem_case.has_device_cuda_mem() ? kDeviceMemory
                                                  : kHostMemory),
      device_id_(mem_case.has_device_cuda_mem()
                     ? mem_case.device_cuda_mem().device_id()
                     : 0) {}

size_t MemoryDevice::Size() const {
  if (memory_type_ == kDeviceMemory) {
    return MemoryDeviceMgr::Singleton()->dev_mem_size();
  } else {
    return MemoryDeviceMgr::Singleton()->host_mem_size();
  }
}

}  // namespace oneflow
