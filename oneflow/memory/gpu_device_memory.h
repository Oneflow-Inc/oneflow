#ifndef _MEMORY_GPU_DEVICE_MEMORY_H_
#define _MEMORY_GPU_DEVICE_MEMORY_H_

#include <glog/logging.h>
#include "device/device_alternate.h"
#include <cuda_runtime.h>
#include <stdio.h>

namespace oneflow {
// GPU Memory Interface Implementation
class GPUDeviceMemory {
 public:
  inline static void* Alloc(size_t size);

  inline static void Free(void* ptr);

 private:
  static const size_t alignment_ = 256; 
};

inline void* GPUDeviceMemory::Alloc(size_t size) {
  void* ret = nullptr;
  CUDA_CHECK(cudaMalloc(&ret, size));
  return ret;
}

inline void GPUDeviceMemory::Free(void* ptr) {
  CUDA_CHECK(cudaFree(ptr));
}
}  // namespace oneflow
#endif  // _MEMORY_GPU_DEVICE_MEMORY_H_
