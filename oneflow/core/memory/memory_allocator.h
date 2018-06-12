#ifndef ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class MemoryAllocator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryAllocator);
  ~MemoryAllocator();

  char* Allocate(MemoryCase mem_case, std::size_t size);

 private:
  friend class Global<MemoryAllocator>;

  MemoryAllocator() = default;
  void Deallocate(char* dptr, MemoryCase mem_case);

  std::list<std::function<void()>> deleters_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
