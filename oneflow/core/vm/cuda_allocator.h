#ifndef ONEFLOW_CORE_VM_CUDA_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_CUDA_ALLOCATOR_H_

#include <cstdint>
#include "oneflow/core/vm/allocator.h"

namespace oneflow {
namespace vm {

class CudaAllocator final : public Allocator {
 public:
  explicit CudaAllocator(int64_t device_id) : Allocator(), device_id_(device_id) {}
  ~CudaAllocator() override = default;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;

 private:
  int64_t device_id_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_ALLOCATOR_H_
