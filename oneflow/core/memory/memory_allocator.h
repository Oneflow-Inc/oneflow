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

inline bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  if (lhs.has_host_mem() && rhs.has_host_mem()) {
    const HostMemory& lhs_host_mem = lhs.host_mem();
    const HostMemory& rhs_host_mem = rhs.host_mem();
    if (lhs_host_mem.has_cuda_pinned_mem() && rhs_host_mem.has_cuda_pinned_mem()) {
      return lhs_host_mem.cuda_pinned_mem().device_id()
             == rhs_host_mem.cuda_pinned_mem().device_id();
    } else {
      return (!lhs_host_mem.has_cuda_pinned_mem()) && (!rhs_host_mem.has_cuda_pinned_mem());
    }
  }
  if (lhs.has_device_cuda_mem() && rhs.has_device_cuda_mem()) {
    return lhs.device_cuda_mem().device_id() == rhs.device_cuda_mem().device_id();
  }
  return false;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
