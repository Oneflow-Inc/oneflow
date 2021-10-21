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
#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_ALLOCATOR_CORE_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_ALLOCATOR_CORE_H_

#include <mutex>
#include <thread>
#include "oneflow/core/object_msg/object_msg.h"

namespace oneflow {

class ObjMsgChunk;

struct ObjMsgMemBlock final {
 public:
  static ObjMsgMemBlock* PlacementNew(char* mem_ptr, ObjMsgChunk* obj_msg_chunk);

  ObjMsgChunk* mut_chunk() { return chunk_; }

  char* mem_ptr() { return &mem_ptr_[0]; }

  template<typename Enabled = void>
  static constexpr int MemPtrOffset() {
    return offsetof(ObjMsgMemBlock, mem_ptr_);
  }

  ObjMsgChunk* chunk_;
  char mem_ptr_[0];
};

class ObjMsgSizedMemPool;

// clang-format off
OBJECT_MSG_BEGIN(ObjMsgChunk);
 public:
  void __Init__(ObjMsgSizedMemPool* mem_pool, int64_t mem_size);
  void __Delete__();

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, mem_size);
  OBJECT_MSG_DEFINE_PTR(ObjMsgMemBlock, mem_block);
  OBJECT_MSG_DEFINE_PTR(ObjMsgSizedMemPool, mem_pool);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(list);

OBJECT_MSG_END(ObjMsgChunk);
// clang-format on

using ObjMsgChunkList = OBJECT_MSG_LIST(ObjMsgChunk, list);

// clang-format off
OBJECT_MSG_BEGIN(ObjMsgSizedMemPool);
 public:
  void __Init__(int64_t fiexed_mem_size, int64_t prefetch_cnt);
  void __Delete__();

  char* Allocate(std::mutex* mutex);
  void Deallocate(std::mutex* mutex, char* ptr);

  // fields
  OBJECT_MSG_DEFINE_LIST_HEAD(ObjMsgChunk, list, occupied_chunk_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(ObjMsgChunk, list, free_chunk_list);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, prefetch_cnt);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, mem_block_size);

 private:
  char* Allocate();
  void Deallocate(char* ptr);
  void Prefetch();
  void Prefetch(ObjMsgChunkList* free_list);
  void AppendToFreeList(ObjMsgChunkList* free_list);
OBJECT_MSG_END(ObjMsgSizedMemPool);
// clang-format on

class CachedObjectMsgAllocatorBase : public ObjectMsgAllocator {
 public:
  CachedObjectMsgAllocatorBase(const CachedObjectMsgAllocatorBase&) = delete;
  CachedObjectMsgAllocatorBase(CachedObjectMsgAllocatorBase&&) = delete;
  CachedObjectMsgAllocatorBase(ObjectMsgAllocator* backend_allocator, int64_t mem_size_shift_max,
                               int64_t prefetch_cnt)
      : backend_allocator_(backend_allocator),
        mem_size_shift_max_(mem_size_shift_max),
        prefetch_cnt_(prefetch_cnt) {
    Prefetch();
  }

  char* RoundUpAllocate(std::mutex* mutex, std::size_t size);
  void RoundUpDeallocate(std::mutex* mutex, char* ptr, std::size_t size);

 private:
  int RoundUpShift(std::size_t size) const;

  static const std::size_t kMemSizeShiftMin = 6;
  void Prefetch();

  ObjectMsgAllocator* backend_allocator_;
  std::size_t mem_size_shift_max_;
  int64_t prefetch_cnt_;
  std::vector<ObjectMsgPtr<ObjMsgSizedMemPool>> allocators_;
};

class CachedObjectMsgAllocator : public CachedObjectMsgAllocatorBase {
 public:
  CachedObjectMsgAllocator(const CachedObjectMsgAllocator&) = delete;
  CachedObjectMsgAllocator(CachedObjectMsgAllocator&&) = delete;

  CachedObjectMsgAllocator(int mem_size_shift_max, int64_t prefetch_cnt)
      : CachedObjectMsgAllocatorBase(ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator(),
                                     mem_size_shift_max, prefetch_cnt) {}
  CachedObjectMsgAllocator(ObjectMsgAllocator* backend_allocator, int64_t mem_size_shift_max,
                           int64_t prefetch_cnt)
      : CachedObjectMsgAllocatorBase(backend_allocator, mem_size_shift_max, prefetch_cnt) {}

  ~CachedObjectMsgAllocator() override = default;

  char* Allocate(std::size_t size) override { return RoundUpAllocate(&mutex_, size); }
  void Deallocate(char* ptr, std::size_t size) override {
    return RoundUpDeallocate(&mutex_, ptr, size);
  }

 private:
  std::mutex mutex_;
};

class ThreadUnsafeObjectMsgAllocator : public CachedObjectMsgAllocatorBase {
 public:
  ThreadUnsafeObjectMsgAllocator(const ThreadUnsafeObjectMsgAllocator&) = delete;
  ThreadUnsafeObjectMsgAllocator(ThreadUnsafeObjectMsgAllocator&&) = delete;

  ThreadUnsafeObjectMsgAllocator(int mem_size_shift_max, int64_t prefetch_cnt)
      : CachedObjectMsgAllocatorBase(ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator(),
                                     mem_size_shift_max, prefetch_cnt),
        thread_id_(std::this_thread::get_id()) {}
  ThreadUnsafeObjectMsgAllocator(ObjectMsgAllocator* backend_allocator, int64_t mem_size_shift_max,
                                 int64_t prefetch_cnt)
      : CachedObjectMsgAllocatorBase(backend_allocator, mem_size_shift_max, prefetch_cnt),
        thread_id_(std::this_thread::get_id()) {}

  ~ThreadUnsafeObjectMsgAllocator() override { CHECK(thread_id_ == std::this_thread::get_id()); }
  char* Allocate(std::size_t size) override;
  void Deallocate(char* ptr, std::size_t size) override;

 private:
  std::thread::id thread_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_ALLOCATOR_CORE_H_
