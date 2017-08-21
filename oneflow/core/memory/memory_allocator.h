#ifndef ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class MemoryAllocator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryAllocator);
  ~MemoryAllocator() = default;

  OF_SINGLETON(MemoryAllocator);

  std::tuple<char*, std::function<void()>, void*> Allocate(MemoryCase mem_case,
                                                           std::size_t size);

 private:
  MemoryAllocator() = default;
  void Deallocate(char* dptr, MemoryCase mem_case);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
