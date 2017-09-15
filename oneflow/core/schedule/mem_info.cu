#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/schedule/mem_info.h"

namespace oneflow {
namespace schedule {

uint32_t MemInfo::GetGpuNum() const {
  int num = 0;
  CudaCheck(cudaGetDeviceCount(&num));
  return static_cast<uint32_t>(num);
}

uint64_t MemInfo::GetGpuRamSize(int32_t dev) const {
  cudaSetDevice(dev);
  cudaDeviceProp gpu_prop;
  cudaGetDeviceProperties(&gpu_prop, dev);
  return static_cast<uint64_t>(gpu_prop.totalGlobalMem);
}

}  // namespace schedule
}  // namespace oneflow
