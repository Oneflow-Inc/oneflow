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
#include <iostream>
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {

ObjMsgMemBlock* ObjMsgMemBlock::PlacementNew(char* mem_ptr, ObjMsgChunk* obj_msg_chunk) {
  ObjMsgMemBlock* ret = new (mem_ptr) ObjMsgMemBlock();
  ret->chunk_ = obj_msg_chunk;
  return ret;
}

void ObjMsgChunk::__Init__(ObjMsgSizedMemPool* mem_pool, int64_t mem_size) {
  set_mem_size(mem_size);
  char* mem_ptr = mut_allocator()->Allocate(sizeof(ObjMsgMemBlock) + mem_size);
  set_mem_block(ObjMsgMemBlock::PlacementNew(mem_ptr, this));
  set_mem_pool(mem_pool);
}

void ObjMsgChunk::__Delete__() {
  mut_allocator()->Deallocate(reinterpret_cast<char*>(mutable_mem_block()),
                              sizeof(ObjMsgMemBlock) + this->mem_size());
}

void ObjMsgSizedMemPool::__Init__(int64_t fixed_mem_size, int64_t prefetch_cnt) {
  set_mem_block_size(fixed_mem_size);
  set_prefetch_cnt(prefetch_cnt);
  Prefetch();
}

void ObjMsgSizedMemPool::__Delete__() { CHECK(occupied_chunk_list().empty()); }

char* ObjMsgSizedMemPool::Allocate(std ::mutex* mutex) {
  if (mutex == nullptr) { return Allocate(); }
  std::unique_lock<std::mutex> lock(*mutex);
  return Allocate();
}

char* ObjMsgSizedMemPool::Allocate() {
  if (free_chunk_list().empty()) { Prefetch(); }
  auto chunk = mut_free_chunk_list()->MoveFrontToDstBack(mut_occupied_chunk_list());
  return chunk->mutable_mem_block()->mem_ptr();
}

void ObjMsgSizedMemPool::Deallocate(std ::mutex* mutex, char* ptr) {
  if (mutex == nullptr) { return Deallocate(ptr); }
  std::unique_lock<std::mutex> lock(*mutex);
  return Deallocate(ptr);
}

void ObjMsgSizedMemPool::Deallocate(char* ptr) {
  auto* chunk =
      StructField<ObjMsgMemBlock, char, ObjMsgMemBlock::MemPtrOffset()>::StructPtr4FieldPtr(ptr)
          ->mut_chunk();
  mut_occupied_chunk_list()->MoveToDstBack(chunk, mut_free_chunk_list());
}

void ObjMsgSizedMemPool::Prefetch() {
  ObjMsgChunkList free_list;
  Prefetch(&free_list);
  AppendToFreeList(&free_list);
}

void ObjMsgSizedMemPool::Prefetch(ObjMsgChunkList* free_list) {
  for (int64_t i = 0; i < prefetch_cnt(); ++i) {
    auto chunk = ObjectMsgPtr<ObjMsgChunk>::NewFrom(mut_allocator(), this, mem_block_size());
    free_list->PushBack(chunk.Mutable());
  }
}

void ObjMsgSizedMemPool::AppendToFreeList(ObjMsgChunkList* free_list) {
  free_list->MoveTo(mut_free_chunk_list());
}

int CachedObjectMsgAllocatorBase::RoundUpShift(std::size_t size) const {
  return std::max<int>(kMemSizeShiftMin, (static_cast<int>(std::log2(size)) + 1));
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
    auto mem_pool =
        ObjectMsgPtr<ObjMsgSizedMemPool>::NewFrom(backend_allocator_, 1 << i, prefetch_cnt_);
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
