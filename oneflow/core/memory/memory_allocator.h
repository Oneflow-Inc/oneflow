#ifndef ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

class MemoryAllocator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryAllocator);
  ~MemoryAllocator();

  char* Allocate(MemoryCase mem_case, std::size_t size);
  template<typename T>
  T* PlacementNew(T* mem_ptr);

 private:
  friend class Global<MemoryAllocator>;

  MemoryAllocator() = default;
  void Deallocate(char* dptr, MemoryCase mem_case);

  std::mutex deleters_mutex_;
  std::list<std::function<void()>> deleters_;
};

template<typename T>
T* MemoryAllocator::PlacementNew(T* mem_ptr) {
  T* obj = new (mem_ptr) T();
  {
    std::unique_lock<std::mutex> lock(deleters_mutex_);
    deleters_.push_front([obj] { obj->~T(); });
  }
  CHECK_EQ(mem_ptr, obj);
  return obj;
}

struct MemoryAllocatorImpl final {
  static void* Allocate(MemoryCase mem_case, size_t size);
  static void Deallocate(void* ptr, MemoryCase mem_case);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
