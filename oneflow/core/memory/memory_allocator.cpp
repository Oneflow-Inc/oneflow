#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

std::tuple<char*, void*, std::function<void()>> MemoryAllocator::Allocate(MemoryCase mem_case,
                                                                          std::size_t size) {
  const int memset_val = 0;
  char* dptr = nullptr;
  void* comm_net_token = nullptr;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().used_by_device()) {
      CudaCheck(cudaMallocHost(&dptr, size));
    } else {
      dptr = reinterpret_cast<char*>(malloc(size));
      CHECK_NOTNULL(dptr);
    }
    if (mem_case.host_mem().used_by_network()) {
      comm_net_token = Global<CommNet>::Get()->RegisterMemory(dptr, size);
    }
    memset(dptr, memset_val, size);
  } else if (mem_case.has_device_cuda_mem()) {
    int32_t current_device_id;
    CudaCheck(cudaGetDevice(&current_device_id));
    CHECK_EQ(mem_case.device_cuda_mem().device_id(), current_device_id);
    CudaCheck(cudaMalloc(&dptr, size));
    CudaCheck(cudaMemset(dptr, memset_val, size));
  } else {
    UNIMPLEMENTED();
  }
  return std::make_tuple(
      dptr, comm_net_token,
      std::bind(&MemoryAllocator::Deallocate, this, dptr, comm_net_token, mem_case));
}

void MemoryAllocator::Deallocate(char* dptr, void* comm_net_token, MemoryCase mem_case) {
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().used_by_network()) {
      Global<CommNet>::Get()->UnRegisterMemory(comm_net_token);
    }
    if (mem_case.host_mem().used_by_device()) {
      CudaCheck(cudaFreeHost(dptr));
    } else {
      free(dptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
    int32_t current_device_id = -1;
    CudaCheck(cudaGetDevice(&current_device_id));
    CHECK_EQ(mem_case.device_cuda_mem().device_id(), current_device_id);
    CudaCheck(cudaFree(dptr));
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
