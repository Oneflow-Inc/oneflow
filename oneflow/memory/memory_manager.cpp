#include "memory/memory_manager.h"

namespace oneflow {

std::pair<void*, std::function<void(void*)>> MemoryMgr::AllocateMem(
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

void MemoryMgr::DeallocateMem(void* dptr, MemoryCase mem_case) {
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

} // namespace oneflow
