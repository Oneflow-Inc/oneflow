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
  template<typename T>
  T* New();

 private:
  friend class Global<MemoryAllocator>;

  MemoryAllocator() = default;
  void Deallocate(char* dptr, MemoryCase mem_case);

  std::mutex deleters_mutex_;
  std::list<std::function<void()>> deleters_;
};

template<typename T>
T* MemoryAllocator::New() {
  T* obj = new T();
  {
    std::unique_lock<std::mutex> lck(deleters_mutex_);
    deleters_.push_front([obj] { delete obj; });
  }
  return obj;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
