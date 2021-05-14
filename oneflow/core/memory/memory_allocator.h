/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

// 内存管理类，会保存待释放的内存信息
class MemoryAllocator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryAllocator);
  MemoryAllocator() = default;
  ~MemoryAllocator();

  char* Allocate(MemoryCase mem_case, std::size_t size);
  template<typename T>
  T* PlacementNew(T* mem_ptr);

 private:
  void Deallocate(char* dptr, MemoryCase mem_case);

  std::mutex deleters_mutex_;
  // 双向链表，保存所有释放内存的函数
  std::list<std::function<void()>> deleters_;
};

class Blob;
void InitNonPODTypeBlobIfNeed(MemoryAllocator* allocator, Blob* blob_ptr);

// 在mem_ptr地址初始化T实例，并保存析构函数
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
  static void* AllocateUnPinnedHostMem(size_t size);
  static void DeallocateUnPinnedHostMem(void* ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_ALLOCATOR_H_
