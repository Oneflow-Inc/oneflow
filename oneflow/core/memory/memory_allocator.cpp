#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

MemoryAllocator::~MemoryAllocator() {
  for (std::function<void()> deleter : deleters_) { deleter(); }
}

char* MemoryAllocator::Allocate(MemoryCase mem_case, std::size_t size) {
  const int memset_val = 0;
  char* dptr = nullptr;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      if (Global<JobDesc>::Get()->enable_numa_aware_cuda_malloc_host()) {
        NumaAwareCudaMallocHost(mem_case.host_mem().cuda_pinned_mem().device_id(), &dptr, size);
      } else {
        CudaCheck(cudaMallocHost(&dptr, size));
      }
    } else {
      dptr = reinterpret_cast<char*>(malloc(size));
      CHECK_NOTNULL(dptr);
    }
    memset(dptr, memset_val, size);
  } else if (mem_case.has_device_cuda_mem()) {
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    CudaCheck(cudaMalloc(&dptr, size));
    CudaCheck(cudaMemset(dptr, memset_val, size));
  } else {
    UNIMPLEMENTED();
  }
  deleters_.push_front(std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case));
  return dptr;
}

void MemoryAllocator::Deallocate(char* dptr, MemoryCase mem_case) {
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      CudaCheck(cudaFreeHost(dptr));
    } else {
      free(dptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    CudaCheck(cudaFree(dptr));
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
