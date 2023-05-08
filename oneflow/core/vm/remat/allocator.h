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
#ifndef ONEFLOW_CORE_VM_DTR_EP_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_DTR_EP_ALLOCATOR_H_

#include <cstdint>
#include "oneflow/core/common/env_var/remat.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/common/util.h"
#include "nlohmann/json.hpp"
#include "oneflow/core/vm/thread_safe_guard.h"

namespace oneflow {

namespace vm {

class EagerBlobObject;
class RematableTensorStorage;

class RematEpAllocator final : public Allocator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RematEpAllocator);
  explicit RematEpAllocator(size_t alignment, std::unique_ptr<Allocator>&& backend);
  ~RematEpAllocator() override;
  void DeviceReset() override;

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
  void LinkStorageAndPtr(RematableTensorStorage* storage, const char* mem_ptr);
  void CheckPieces();
  void DisplayAllPieces();
  size_t allocated_memory();
  void set_left(bool is_left) { left = is_left; }
  bool left = true;

  size_t iterate_group_index(bool high) const;

  bool first_time = true;

 private:
  const size_t alignment_;
  const std::unique_ptr<Allocator> backend_;
  ReentrantThreadSafeLock thread_lock_;

  using offset_t = size_t;

  offset_t get_offset(const char* mem_ptr) const;

  // Piece is the basic memory unit of CudaAllocator.
  // A Piece is either is free(is_free = true) or in used(is_free = false).
  // Pieces are stored in a linked list. The Piece's prev and next are
  // continuous with the current Piece in physical memory.
  struct Piece {
    size_t size = 0;
    char* ptr = nullptr;
    bool is_free = true;
    Piece* prev = nullptr;
    Piece* next = nullptr;
    vm::RematableTensorStorage* tensor = nullptr;
    bool is_left = true;
  };

  Maybe<bool> InSmallMemoryArea(void* ptr);

  offset_t FindProperPositionInGroup(Piece* piece, size_t group_idx, size_t request_size) const;

  Piece* AllocateMemoryInPiece(Piece* piece, offset_t offset_in_piece, size_t size);

  void InsertToFreeList(Piece* piece);
  void EraseFromFreeList(Piece* piece);

  void InitMemory();

  // Try find free Piece which size is larger than aligned_size
  // Return nullptr when find failure
  Maybe<Piece*> FindPiece(size_t aligned_size, bool after_eviction);
  void Display();

  // Create new empty Piece or recycle a Piece from recycle_piece_list_
  Piece* AllocatePiece();
  // Delete a Piece and move in the linked list recycle_piece_list_
  void DeallocatePiece(Piece* piece);

  // Insert a {piece->ptr, piece} pair into the ptr2piece_ map for search Piece when call
  // Deallocate()
  void InsertPiece2PtrMap(Piece* piece);
  // Erase the {piece->ptr, piece} pair from ptr2piece_ because the ptr is useless
  // Usually call before DeallocatePiece()
  void ErasePieceFromPtrMap(Piece* piece);

  void MergeNeighbourFreePiece(Piece* lhs, Piece* rhs);

  Maybe<Piece*> EvictAndFindPieceOnce(size_t required_size);
  Maybe<Piece*> EvictAndFindPieceLoop(size_t required_size, bool consider_neighbor);

  char* memory_ = nullptr;
  size_t memory_size_;
  void* small_piece_area_ptr_ = nullptr;

  // hold the lifetime of Piece
  std::vector<std::unique_ptr<Piece>> pieces_;
  struct PieceCmp {
    bool operator()(const Piece* lhs, const Piece* rhs) const {
      if (lhs->size != rhs->size) { return lhs->size < rhs->size; }
      // compare pointer by raw < or > is undefined behavior
      return std::less<>{}(lhs->ptr, rhs->ptr);
    }
  };
  std::vector<std::set<Piece*, PieceCmp>> free_pieces_overlapping_with_group_;
  // std::map is sorted by key, so we can find contiguous memory by it
  std::map<const char*, Piece*> ptr2piece_;
  Piece* recycle_piece_list_;
  size_t total_allocate_bytes_ = 0;
  size_t total_deallocate_bytes_ = 0;

  // -----
  size_t normal_group_num_;
  std::vector<size_t> group_indexes_;
  mutable size_t cur_group_index_id_;
  mutable size_t cur_group_index_id_high_cost_;
  bool enable_left_and_right_;
  std::vector<std::pair<offset_t, offset_t>> group_boundaries_;

  size_t group_index(bool high) const;
};

class DtrEpAllocatorProxy final : public Allocator {
 public:
  explicit DtrEpAllocatorProxy(vm::RematEpAllocator* allocator) : allocator(allocator) {}
  void DeviceReset() override { allocator->DeviceReset(); }

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override {
    return allocator->Allocate(mem_ptr, size);
  }
  void Deallocate(char* mem_ptr, std::size_t size) override {
    allocator->Deallocate(mem_ptr, size);
  }
  vm::RematEpAllocator* const allocator;
};

}  // namespace vm

namespace remat {
class AllocatorManager {
 public:
  vm::RematEpAllocator* CreateOrGetAllocator(DeviceType device_type, size_t device_index);

 private:
  std::unordered_map<std::pair<DeviceType, size_t>, std::unique_ptr<vm::RematEpAllocator>>
      allocators_;
};

}  // namespace remat
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_DTR_EP_ALLOCATOR_H_
