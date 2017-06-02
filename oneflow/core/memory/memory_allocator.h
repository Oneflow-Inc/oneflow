#ifndef ONEFLOW_MEMORY_MEMORY_ALLOCATOR_H_
#define ONEFLOW_MEMORY_MEMORY_ALLOCATOR_H_

#include "oneflow/core/memory/memory_case.pb.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class MemoryAllocator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryAllocator);
  ~MemoryAllocator() = default;
  
  static MemoryAllocator& Singleton() {
    static MemoryAllocator obj;
    return obj;
  }

  std::pair<char*, std::function<void()>> Allocate(
      MemoryCase mem_case, std::size_t size);

 private:
  MemoryAllocator() = default;
  void Deallocate(char* dptr, MemoryCase mem_case);

};

} // namespace oneflow

#endif // ONEFLOW_MEMORY_MEMORY_ALLOCATOR_H_
