#include "oneflow/core/memory/dynamic_memory_allocator.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

void* DynamicMemoryAllocator::New(MemoryCase mem_case, size_t size) {
  void* ptr = nullptr;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      if (Global<ResourceDesc>::Get()->enable_numa_aware_cuda_malloc_host()) {
        NumaAwareCudaMallocHost(mem_case.host_mem().cuda_pinned_mem().device_id(), &ptr, size);
      } else {
        CudaCheck(cudaMallocHost(&ptr, size));
      }
    } else {
      ptr = malloc(size);
      CHECK_NOTNULL(ptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    CudaCheck(cudaMalloc(&ptr, size));
  } else {
    UNIMPLEMENTED();
  }
  return ptr;
}

void DynamicMemoryAllocator::Delete(void* ptr, MemoryCase mem_case) {
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      CudaCheck(cudaFreeHost(ptr));
    } else {
      free(ptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    CudaCheck(cudaFree(ptr));
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
