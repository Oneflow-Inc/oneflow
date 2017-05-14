#ifndef ONEFLOW_MEMORY_MEMORY_MANAGER_H_
#define ONEFLOW_MEMORY_MEMORY_MANAGER_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "common/util.h"

namespace oneflow {

enum class MemoryType {
  kHostPageableMemory = 0,
  kHostPinnedMemory,
  kDeviceGPUMemory
};

struct MemoryCase {
  MemoryType type;
  int32_t device_id;
};

class MemoryMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryMgr);
  ~MemoryMgr() = default;
  
  static MemoryMgr& Singleton() {
    static MemoryMgr obj;
    return obj;
  }

  std::pair<void*, std::function<void(void*)>> AllocateMem(
      MemoryCase mem_case, std::size_t size);

 private:
  MemoryMgr();
  void DeallocateMem(void* dptr, MemoryCase mem_case);

};

} // namespace oneflow

#endif // ONEFLOW_MEMORY_MEMORY_MANAGER_H_
