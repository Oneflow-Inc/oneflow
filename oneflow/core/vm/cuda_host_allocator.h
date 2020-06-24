#ifndef ONEFLOW_CORE_VM_CUDA_HOST_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_CUDA_HOST_ALLOCATOR_H_

#include <cstdint>
#include "oneflow/core/vm/allocator.h"

namespace oneflow {
namespace vm {

class CudaHostAllocator final : public Allocator {
 public:
  explicit CudaHostAllocator() : Allocator() {}
  ~CudaHostAllocator() override = default;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_HOST_ALLOCATOR_H_
