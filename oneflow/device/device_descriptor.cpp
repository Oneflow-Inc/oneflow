#include "device/device_descriptor.h"
#include "device/device_alternate.h"

namespace oneflow {
DeviceDescriptor::DeviceDescriptor(int32_t physical_id)
  : physical_id_(physical_id) {
  GetDeviceProperties();
}
void DeviceDescriptor::GetDeviceProperties() {
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, physical_id_));
  name_ = device_prop.name;
  compute_capability_major_ = device_prop.major;
  compute_capability_minor_ = device_prop.minor;

  multi_processor_count_ = device_prop.multiProcessorCount;
  cuda_cores_per_multi_processor_ = _ConvertSMVer2Cores(
    compute_capability_major_,
    compute_capability_minor_);
  total_cuda_cores_ = cuda_cores_per_multi_processor_
    * multi_processor_count_;
  clock_rate_ = device_prop.clockRate / 1000;

  max_threads_per_multi_processor_ = device_prop.maxThreadsPerMultiProcessor;
  warp_size_ = device_prop.warpSize;
  max_threads_per_block_ = device_prop.maxThreadsPerBlock;
  max_threads_dim_[0] = device_prop.maxThreadsDim[0];
  max_threads_dim_[1] = device_prop.maxThreadsDim[1];
  max_threads_dim_[2] = device_prop.maxThreadsDim[2];
  max_grid_size_[0] = device_prop.maxGridSize[0];
  max_grid_size_[1] = device_prop.maxGridSize[1];
  max_grid_size_[2] = device_prop.maxGridSize[2];

  concurrent_kernels_ = device_prop.concurrentKernels;
  async_engine_count_ = device_prop.asyncEngineCount;
  total_global_mem_ = device_prop.totalGlobalMem;
  memory_clock_rate_ = device_prop.memoryClockRate / 1000;
  memory_bus_width_ = device_prop.memoryBusWidth;
  return;
}
}  // namespace oneflow
