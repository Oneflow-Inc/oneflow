#ifndef ONEFLOW_CORE_DYNAMIC_MEMORY_MEMORY_ALLOCATOR_H_
#define ONEFLOW_CORE_DYNAMIC_MEMORY_MEMORY_ALLOCATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

struct DynamicMemoryAllocator final {
  static void* New(MemoryCase mem_case, size_t size);
  static void Delete(void* ptr, MemoryCase mem_case);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DYNAMIC_MEMORY_MEMORY_ALLOCATOR_H_
