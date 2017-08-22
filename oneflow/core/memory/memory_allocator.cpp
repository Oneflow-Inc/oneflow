#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/network/network.h"

namespace oneflow {

std::tuple<char*, std::function<void()>, void*> MemoryAllocator::Allocate(
    MemoryCase mem_case, std::size_t size) {
  const int memset_val = 255;
  char* dptr = nullptr;
  void* net_memory_ptr = nullptr;
  if (mem_case.has_host_pageable_mem()) {
    dptr = (char*)malloc(size);
    CHECK(dptr != nullptr);
    memset(dptr, memset_val, size);
  } else if (mem_case.has_host_pinned_mem()) {
    if (mem_case.host_pinned_mem().need_cuda()) {
      CudaCheck(cudaMallocHost(&dptr, size));
    } else {
      dptr = (char*)malloc(size);
      CHECK(dptr != nullptr);
      memset(dptr, memset_val, size);
    }
    if (mem_case.host_pinned_mem().need_rdma()) {
      // TODO();
      Network* net = GetRdmaInstance();
      net_memory_ptr = net->RegisterMemory(dptr, size);
    }
    memset(dptr, memset_val, size);
  } else if (mem_case.has_device_cuda_mem()) {
    int32_t current_device_id;
    CudaCheck(cudaGetDevice(&current_device_id));
    CHECK_EQ(mem_case.device_cuda_mem().device_id(), current_device_id);
    CudaCheck(cudaMalloc(&dptr, size));
    CudaCheck(cudaMemset(dptr, memset_val, size));
  } else {
    UNEXPECTED_RUN();
  }
  std::tuple<char*, std::function<void()>, void*> result{
      dptr, std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case),
      net_memory_ptr};
  return result;
}

void MemoryAllocator::Deallocate(char* dptr, MemoryCase mem_case) {
  if (mem_case.has_host_pageable_mem()) {
    free(dptr);
  } else if (mem_case.has_host_pinned_mem()) {
    if (mem_case.host_pinned_mem().need_rdma()) {
      Network* net = GetRdmaInstance();
      net->UnRegisterMemory(dptr);
    }
    if (mem_case.host_pinned_mem().need_cuda()) {
      CudaCheck(cudaFreeHost(&dptr));
    } else {
      free(dptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
    int32_t current_device_id;
    CudaCheck(cudaGetDevice(&current_device_id));
    CHECK_EQ(mem_case.device_cuda_mem().device_id(), current_device_id);
    CudaCheck(cudaFree(&dptr));
  }
}

}  // namespace oneflow
