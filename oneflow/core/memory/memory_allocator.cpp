#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/comm_network/data_comm_network.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

std::tuple<char*, const void*, std::function<void()>> MemoryAllocator::Allocate(
    MemoryCase mem_case, std::size_t size) {
  const int memset_val = 255;
  char* dptr = nullptr;
  const void* comm_net_token = nullptr;
  if (mem_case.has_host_pageable_mem()) {
    dptr = (char*)malloc(size);
    CHECK_NOTNULL(dptr);
    memset(dptr, memset_val, size);
  } else if (mem_case.has_host_pinned_mem()) {
    if (mem_case.host_pinned_mem().used_by_device()) {
      CudaCheck(cudaMallocHost(&dptr, size));
    } else {
      dptr = (char*)malloc(size);
    }
    if (mem_case.host_pinned_mem().used_by_network()) {
      comm_net_token = DataCommNet::Singleton()->RegisterMemory(dptr);
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
  return std::make_tuple(dptr, comm_net_token,
                         std::bind(&MemoryAllocator::Deallocate, this, dptr,
                                   comm_net_token, mem_case));
}

void MemoryAllocator::Deallocate(char* dptr, const void* comm_net_token,
                                 MemoryCase mem_case) {
  if (mem_case.has_host_pageable_mem()) {
    free(dptr);
  } else if (mem_case.has_host_pinned_mem()) {
    if (mem_case.host_pinned_mem().used_by_network()) {
      DataCommNet::Singleton()->UnRegisterMemory(comm_net_token);
    }
    if (mem_case.host_pinned_mem().used_by_device()) {
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
