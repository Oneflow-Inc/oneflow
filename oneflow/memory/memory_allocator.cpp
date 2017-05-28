#include "memory/memory_allocator.h"

namespace oneflow {

std::pair<char*, std::function<void()>> MemoryAllocator::Allocate(
    MemoryCase mem_case,std::size_t size) {
  TODO();
  /*
  char* dptr = nullptr;
  if (mem_case.has_host_pageable_mem()) {
    dptr = (char*) malloc (size);
    CHECK(dptr != nullptr);
  } else if (mem_case.has_cuda_pinned_mem()) {
    CHECK_EQ(cudaMallocHost(&dptr, size), 0);
  } else if (mem_case.has_rdma_pinned_mem()) {
    TODO();
  } else if (mem_case.has_gpu_mem()) {
    int32_t current_device_id;
    CHECK_EQ(cudaGetDevice(&current_device_id), 0);
    CHECK_EQ(cudaSetDevice(mem_case.gpu_mem().device_id()), 0);
    CHECK_EQ(cudaMalloc(&dptr, size), 0);
    CHECK_EQ(cudaSetDevice(current_device_id), 0);
  }
  return {dptr, std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case)};
  */
}

void MemoryAllocator::Deallocate(char* dptr, MemoryCase mem_case) {
  TODO();
  /*
  if (mem_case.has_host_pageable_mem()) {
    free(dptr);
  } else if (mem_case.has_cuda_pinned_mem()) {
    CHECK_EQ(cudaFreeHost(&dptr), 0);
  } else if (mem_case.has_rdma_pinned_mem()) {
    TODO();
  } else if (mem_case.has_gpu_mem()) {
    int32_t current_device_id;
    CHECK_EQ(cudaGetDevice(&current_device_id), 0);
    CHECK_EQ(cudaSetDevice(mem_case.gpu_mem().device_id()), 0);
    CHECK_EQ(cudaFree(&dptr), 0);
    CHECK_EQ(cudaSetDevice(current_device_id), 0);
  }
  */
}

} // namespace oneflow
