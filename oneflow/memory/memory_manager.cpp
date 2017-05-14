#include "memory/memory_manager.h"

namespace oneflow {

std::pair<void*, std::function<void(void*)>> MemoryMgr::Allocate(
    MemoryCase mem_case,std::size_t size) {
  void* dptr = nullptr;
  switch(mem_case.type) {
    case MemoryType::kHostPageableMemory: {
      dptr = malloc(size);
      CHECK(dptr != nullptr);
      break;
    }
    case MemoryType::kHostPinnedMemory: {
      CHECK_EQ(cudaMallocHost(&dptr, size), 0);
      break;
    }
    case MemoryType::kDeviceGPUMemory: {
      int32_t current_device_id;
      CHECK_EQ(cudaGetDevice(&current_device_id), 0);
      CHECK_EQ(cudaSetDevice(mem_case.device_id), 0);
      CHECK_EQ(cudaMalloc(&dptr, size), 0);
      CHECK_EQ(cudaSetDevice(current_device_id), 0);
      break;
    }
  }
  return {dptr, std::bind(&MemoryMgr::Deallocate,
                          this, std::placeholders::_1, mem_case)};
}

void MemoryMgr::Deallocate(void* dptr, MemoryCase mem_case) {
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
      int32_t current_device_id;
      CHECK_EQ(cudaGetDevice(&current_device_id), 0);
      CHECK_EQ(cudaSetDevice(mem_case.device_id), 0);
      CHECK_EQ(cudaFree(&dptr), 0);
      CHECK_EQ(cudaSetDevice(current_device_id), 0);
      break;
    }
  } 
}

} // namespace oneflow
