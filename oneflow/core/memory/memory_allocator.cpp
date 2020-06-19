#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

void* MemoryAllocatorImpl::Allocate(MemoryCase mem_case, size_t size) {
  void* ptr = nullptr;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      if (Global<ResourceDesc, ForSession>::Get()->enable_numa_aware_cuda_malloc_host()) {
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

void MemoryAllocatorImpl::Deallocate(void* ptr, MemoryCase mem_case) {
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

void* MemoryAllocatorImpl::AllocateUnPinnedHostMem(size_t size) {
  void* ptr = malloc(size);
  CHECK_NOTNULL(ptr);
  return ptr;
}

void MemoryAllocatorImpl::DeallocateUnPinnedHostMem(void* ptr) { free(ptr); }

MemoryAllocator::~MemoryAllocator() {
  for (std::function<void()> deleter : deleters_) { deleter(); }
}

char* MemoryAllocator::Allocate(MemoryCase mem_case, std::size_t size) {
  const int memset_val = 0;
  char* dptr = static_cast<char*>(MemoryAllocatorImpl::Allocate(mem_case, size));
  if (mem_case.has_host_mem()) {
    memset(dptr, memset_val, size);
  } else if (mem_case.has_device_cuda_mem()) {
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    CudaCheck(cudaMemset(dptr, memset_val, size));
  } else {
    UNIMPLEMENTED();
  }
  deleters_.push_front(std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case));
  return dptr;
}

void MemoryAllocator::Deallocate(char* dptr, MemoryCase mem_case) {
  MemoryAllocatorImpl::Deallocate(static_cast<void*>(dptr), mem_case);
}

}  // namespace oneflow
