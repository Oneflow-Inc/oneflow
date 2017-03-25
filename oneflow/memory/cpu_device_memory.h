#ifndef _MEMORY_CPU_DEVICE_MEMORY_H_
#define _MEMORY_CPU_DEVICE_MEMORY_H_

#include <glog/logging.h>
#include <stdio.h>

namespace oneflow {
// CPU Memory Interface Implementation
class CPUDeviceMemory {
 public:
  inline static void* Alloc(size_t size);

  inline static void Free(void* ptr);

 private:
  static const size_t alignment_ = 16;
};

inline void* CPUDeviceMemory::Alloc(size_t size) {
#if _MSC_VER
  void* ptr = nullptr;
  ptr = _aligned_malloc(size, alignment_);
  CHECK(ptr != nullptr) << "Cannot allocate memory";
  return ptr;
#else
  CHECK(0) << "Not Implement Error!";
#endif
}

inline void CPUDeviceMemory::Free(void* ptr) {
#if _MSC_VER
  _aligned_free(ptr);
#else
  CHECK(0) << "Not Implement Error!";
#endif
}
}  // namespace oneflow
#endif  // _MEMORY_CPU_DEVICE_MEMORY_H_
