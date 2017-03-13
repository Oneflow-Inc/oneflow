#ifndef _MEMORY_CUDA_PINNED_MEMORY_H_
#define _MEMORY_CUDA_PINNED_MEMORY_H_

#include <glog/logging.h>
#include "device/device_alternate.h"
#include <cuda_runtime.h>

namespace caffe {
// CUDA pinned Memory Interface Implementation
class CUDAPinnedMemory {
 public:
  inline static void* Alloc(size_t size);

  inline static void Free(void* ptr);

 private:
  static const size_t alignment_ = 16;
};

inline void* CUDAPinnedMemory::Alloc(size_t size) {
  void* ret = nullptr;
  CUDA_CHECK(cudaHostAlloc(&ret, size, cudaHostAllocPortable));
  return ret;
}

inline void CUDAPinnedMemory::Free(void* ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}
}  // namespace caffe
#endif  // _MEMORY_CUDA_PINNED_MEMORY_H_
