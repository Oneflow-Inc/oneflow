#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/memory/memory_device.h"

namespace oneflow {

size_t MemoryDeviceMgr::GetThisMachineDeviceMemSize() const {
  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp gpu_prop;
  cudaGetDeviceProperties(&gpu_prop, dev);
  return static_cast<size_t>(gpu_prop.totalGlobalMem);
}

}  // namespace oneflow
