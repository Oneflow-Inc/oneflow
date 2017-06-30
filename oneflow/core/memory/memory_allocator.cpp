#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

std::pair<char*, std::function<void()>> MemoryAllocator::Allocate(
    MemoryCase mem_case, std::size_t size) {
  char* dptr = nullptr;
  if (mem_case.has_host_pageable_mem()) {
    dptr = (char*)malloc(size);
    CHECK(dptr != nullptr);
  } else if (mem_case.has_host_pinned_mem()) {
    if (mem_case.host_pinned_mem().need_cuda()) {
      CHECK_EQ(cudaMallocHost(&dptr, size), 0);
    }
    if (mem_case.host_pinned_mem().need_rdma()) { TODO(); }
  } else if (mem_case.has_device_cuda_mem()) {
    int32_t current_device_id;
    CHECK_EQ(cudaGetDevice(&current_device_id), 0);
    CHECK_EQ(mem_case.device_cuda_mem().device_id(), current_device_id);
    CHECK_EQ(cudaMalloc(&dptr, size), 0);
  }
  return {dptr, std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case)};
}

void MemoryAllocator::Deallocate(char* dptr, MemoryCase mem_case) {
  if (mem_case.has_host_pageable_mem()) {
    free(dptr);
  } else if (mem_case.has_host_pinned_mem()) {
    if (mem_case.host_pinned_mem().need_cuda()) {
      CHECK_EQ(cudaFreeHost(&dptr), 0);
    }
    if (mem_case.host_pinned_mem().need_rdma()) { TODO(); }
  } else if (mem_case.has_device_cuda_mem()) {
    int32_t current_device_id;
    CHECK_EQ(cudaGetDevice(&current_device_id), 0);
    CHECK_EQ(mem_case.device_cuda_mem().device_id(), current_device_id);
    CHECK_EQ(cudaFree(&dptr), 0);
  }
}

}  // namespace oneflow
