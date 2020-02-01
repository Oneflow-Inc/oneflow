#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_ALLOCATOR_CORE_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_ALLOCATOR_CORE_H_

#include "oneflow/core/common/object_msg.h"

namespace oneflow {

class OBJECT_MSG_TYPE(ObjMsgChunk);

struct ObjMsgMemBlock final {
 public:
  static ObjMsgMemBlock* PlacementNew(char* mem_ptr, OBJECT_MSG_TYPE(ObjMsgChunk) * obj_msg_chunk);

  OBJECT_MSG_TYPE(ObjMsgChunk) * mut_chunk() { return chunk_; }

  char* mem_ptr() { return &mem_ptr_[0]; }

  template<typename Enabled = void>
  static constexpr int MemPtrOffset() {
    return (int)(long long)&((ObjMsgMemBlock*)nullptr)->mem_ptr_[0];
  }

  OBJECT_MSG_TYPE(ObjMsgChunk) * chunk_;
  char mem_ptr_[0];
};

class OBJECT_MSG_TYPE(ObjMsgSizedMemPool);

// clang-format off
BEGIN_OBJECT_MSG(ObjMsgChunk);
 public:
  void __Init__(OBJECT_MSG_TYPE(ObjMsgSizedMemPool)* mem_pool, int64_t mem_size);
  void __Delete__();

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, mem_size);
  OBJECT_MSG_DEFINE_RAW_PTR(ObjMsgMemBlock*, mem_block);
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(ObjMsgSizedMemPool)*, mem_pool);

  // links
  OBJECT_MSG_DEFINE_LIST_ITEM(occupied_chunk_list);
  OBJECT_MSG_DEFINE_LIST_ITEM(free_chunk_list);

END_OBJECT_MSG(ObjMsgChunk);
// clang-format on

using FreeObjMsgChunkList = OBJECT_MSG_LIST(ObjMsgChunk, free_chunk_list);

// clang-format off
BEGIN_OBJECT_MSG(ObjMsgSizedMemPool);
 public:
  void __Init__(int64_t fiexed_mem_size, int64_t prefetch_cnt);
  void __Delete__();

  char* Allocate();
  void Deallocate(char* ptr);

  // fields
  OBJECT_MSG_DEFINE_LIST_HEAD(ObjMsgChunk, occupied_chunk_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(ObjMsgChunk, free_chunk_list);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, prefetch_cnt);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(5, int64_t, mem_block_size);

 private:
  void Prefetch();
  void Prefetch(FreeObjMsgChunkList* free_list);
  void AppendToFreeList(FreeObjMsgChunkList* free_list);
END_OBJECT_MSG(ObjMsgSizedMemPool);
// clang-format on

class CachedObjectMsgAllocator final : public ObjectMsgAllocator {
 public:
  CachedObjectMsgAllocator(const CachedObjectMsgAllocator&) = delete;
  CachedObjectMsgAllocator(CachedObjectMsgAllocator&&) = delete;
  CachedObjectMsgAllocator(int mem_size_shift_max, int64_t prefetch_cnt)
      : backend_allocator_(ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator()),
        mem_size_shift_max_(mem_size_shift_max),
        prefetch_cnt_(prefetch_cnt) {
    Prefetch();
  }
  CachedObjectMsgAllocator(ObjectMsgAllocator* backend_allocator, int64_t mem_size_shift_max,
                           int64_t prefetch_cnt)
      : backend_allocator_(backend_allocator),
        mem_size_shift_max_(mem_size_shift_max),
        prefetch_cnt_(prefetch_cnt) {
    Prefetch();
  }
  ~CachedObjectMsgAllocator();

  char* Allocate(std::size_t size) override;
  void Deallocate(char* ptr, std::size_t size) override;

 private:
  static const std::size_t kMemSizeShiftMin = 6;
  void Prefetch();
  static std::size_t RoundUpSize(std::size_t size);

  ObjectMsgAllocator* backend_allocator_;
  std::size_t mem_size_shift_max_;
  int64_t prefetch_cnt_;
  OBJECT_MSG_SKIPLIST(ObjMsgSizedMemPool, mem_block_size) allocators_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_ALLOCATOR_CORE_H_
