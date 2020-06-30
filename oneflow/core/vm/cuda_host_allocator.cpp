#include "oneflow/core/vm/cuda_host_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

void CudaHostAllocator::Allocate(char** mem_ptr, std::size_t size) {
  CudaCheck(cudaMallocHost(mem_ptr, size));
}

void CudaHostAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  CudaCheck(cudaFreeHost(mem_ptr));
}

}  // namespace vm
}  // namespace oneflow
