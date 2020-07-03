#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

void CudaAllocator::Allocate(char** mem_ptr, std::size_t size) {
  cudaSetDevice(device_id_);
  CudaCheck(cudaMalloc(mem_ptr, size));
}

void CudaAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  cudaSetDevice(device_id_);
  CudaCheck(cudaFree(mem_ptr));
}

}  // namespace vm
}  // namespace oneflow
