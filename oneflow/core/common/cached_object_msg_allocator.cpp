#include <iostream>
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {

namespace {

int RoundUpShift(std::size_t size) { return (static_cast<int>(std::log2(size)) + 1); }

}  // namespace

ObjMsgMemBlock* ObjMsgMemBlock::PlacementNew(char* mem_ptr,
                                             OBJECT_MSG_TYPE(ObjMsgChunk) * obj_msg_chunk) {
  ObjMsgMemBlock* ret = new (mem_ptr) ObjMsgMemBlock();
  ret->chunk_ = obj_msg_chunk;
  return ret;
}

void OBJECT_MSG_TYPE(ObjMsgChunk)::__Init__(OBJECT_MSG_TYPE(ObjMsgSizedMemPool) * mem_pool,
                                            int64_t mem_size) {
  set_mem_size(mem_size);
  char* mem_ptr = mut_allocator()->Allocate(sizeof(ObjMsgMemBlock) + mem_size);
  set_mem_block(ObjMsgMemBlock::PlacementNew(mem_ptr, this));
  set_mem_pool(mem_pool);
}

void OBJECT_MSG_TYPE(ObjMsgChunk)::__Delete__() {
  mut_allocator()->Deallocate(reinterpret_cast<char*>(mutable_mem_block()),
                              sizeof(ObjMsgMemBlock) + this->mem_size());
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::__Init__(int64_t fixed_mem_size, int64_t prefetch_cnt) {
  set_mem_block_size(fixed_mem_size);
  set_prefetch_cnt(prefetch_cnt);
  Prefetch();
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::__Delete__() { CHECK(occupied_chunk_list().empty()); }

char* OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::Allocate(std ::mutex* mutex) {
  if (mutex == nullptr) { return Allocate(); }
  std::unique_lock<std::mutex> lock(*mutex);
  return Allocate();
}

char* OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::Allocate() {
  if (free_chunk_list().empty()) { Prefetch(); }
  auto chunk = mut_free_chunk_list()->MoveFrontToDstBack(mut_occupied_chunk_list());
  return chunk->mutable_mem_block()->mem_ptr();
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::Deallocate(std ::mutex* mutex, char* ptr) {
  if (mutex == nullptr) { return Deallocate(ptr); }
  std::unique_lock<std::mutex> lock(*mutex);
  return Deallocate(ptr);
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::Deallocate(char* ptr) {
  auto* chunk =
      StructField<ObjMsgMemBlock, char, ObjMsgMemBlock::MemPtrOffset()>::StructPtr4FieldPtr(ptr)
          ->mut_chunk();
  mut_occupied_chunk_list()->MoveToDstBack(chunk, mut_free_chunk_list());
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::Prefetch() {
  ObjMsgChunkList free_list;
  Prefetch(&free_list);
  AppendToFreeList(&free_list);
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::Prefetch(ObjMsgChunkList* free_list) {
  for (int64_t i = 0; i < prefetch_cnt(); ++i) {
    auto chunk = OBJECT_MSG_PTR(ObjMsgChunk)::New(mut_allocator());
    chunk->__Init__(this, mem_block_size());
    free_list->PushBack(chunk.Mutable());
  }
}

void OBJECT_MSG_TYPE(ObjMsgSizedMemPool)::AppendToFreeList(ObjMsgChunkList* free_list) {
  free_list->MoveTo(mut_free_chunk_list());
}

char* CachedObjectMsgAllocatorBase::RoundUpAllocate(std::mutex* mutex, std::size_t size) {
  return allocators_.at(RoundUpShift(size) - kMemSizeShiftMin)->Allocate(mutex);
}

void CachedObjectMsgAllocatorBase::RoundUpDeallocate(std::mutex* mutex, char* ptr,
                                                     std::size_t size) {
  auto* chunk =
      StructField<ObjMsgMemBlock, char, ObjMsgMemBlock::MemPtrOffset()>::StructPtr4FieldPtr(ptr)
          ->mut_chunk();
  CHECK_LE(size, chunk->mem_size());
  chunk->mutable_mem_pool()->Deallocate(mutex, ptr);
}

void CachedObjectMsgAllocatorBase::Prefetch() {
  CHECK_LE(mem_size_shift_max_, 32);
  CHECK_LT(kMemSizeShiftMin, mem_size_shift_max_);
  allocators_.resize(mem_size_shift_max_ - kMemSizeShiftMin + 1);
  for (int i = kMemSizeShiftMin; i <= mem_size_shift_max_; ++i) {
    auto mem_pool = OBJECT_MSG_PTR(ObjMsgSizedMemPool)::New(backend_allocator_);
    mem_pool->__Init__(1 << i, prefetch_cnt_);
    allocators_.at(i - kMemSizeShiftMin) = mem_pool;
  }
}

char* ThreadUnsafeObjectMsgAllocator::Allocate(std::size_t size) {
  CHECK(thread_id_ == std::this_thread::get_id());
  return RoundUpAllocate(nullptr, size);
}
void ThreadUnsafeObjectMsgAllocator::Deallocate(char* ptr, std::size_t size) {
  CHECK(thread_id_ == std::this_thread::get_id());
  return RoundUpDeallocate(nullptr, ptr, size);
}

}  // namespace oneflow
