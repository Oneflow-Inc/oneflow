#ifndef ONEFLOW_MEMORY_MEMORY_MANAGER_H_
#define ONEFLOW_MEMORY_MEMORY_MANAGER_H_

#include "cuda.h"
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
      MemoryCase mem_case,std::size_t size) {
    switch(mem_case.type) {
      case MemoryType::kHostPageableMemory: {
        dptr = malloc(size);
        CHECK_NE(dptr, NULL);
        break;
      }
      case MemoryType::kHostPinnedMemory: {
        CHECK_EQ(cudaMallocHost(&dptr, size), 0);
        break;
      }
      case MemoryType::kDeviceGPUMemory: {
        CHECK_EQ(cudaSetDevice(mem_case.device_id), 0);
        CHECK_EQ(cudaMalloc(&dptr, size), 0);
        break;
      }
    }
    return {dptr, std::bind(&MemoryMgr::DeallocateMem, this, _1, mem_case)};
  }

 private:
  MemoryMgr();
  
  void DeallocateMem(void* dptr, MemoryCase mem_case) {
    switch(mem_case.type) {
      case MemoryType::kHostPageableMemory: {
        free(dptr);
        break;
      }
      case MemoryType::kHostPinnedMemory: {
        CHECK_EQ(cudaFreeHost(&dptr), 0);
        break;
      }
      case MemoryType::kDeviceGPUMemory: {
        CHECK_EQ(cudaSetDevice(mem_case.device_id), 0);
        CHECK_EQ(cudaFree(&dptr), 0);
        break;
      }
    } 
  }
};

} // namespace oneflow

#endif // ONEFLOW_MEMORY_MEMORY_MANAGER_H_
