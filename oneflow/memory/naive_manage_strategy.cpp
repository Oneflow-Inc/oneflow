#include "memory/naive_manage_strategy.h"
#include "memory/cpu_device_memory.h"
#include "memory/gpu_device_memory.h"
#include "memory/cuda_pinned_memory.h"

namespace caffe {
template <class DeviceMemory>
void* NaiveManageStrategy<DeviceMemory>::Alloc(size_t size) {
  this->alloc_size_ += size;
  return DeviceMemory::Alloc(size);
}

template <class DeviceMemory>
void NaiveManageStrategy<DeviceMemory>::Free(void* ptr, size_t size) {
  this->alloc_size_ -= size;
  DeviceMemory::Free(ptr);
}

template <class DeviceMemory>
size_t NaiveManageStrategy<DeviceMemory>::Size() {
  return this->alloc_size_;
}

template class NaiveManageStrategy<CPUDeviceMemory>;
template class NaiveManageStrategy<GPUDeviceMemory>;
template class NaiveManageStrategy<CUDAPinnedMemory>;
}  // namespace caffe
