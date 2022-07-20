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
#ifndef ONEFLOW_CORE_VM_DTR_CUDA_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_DTR_CUDA_ALLOCATOR_H_

#include <cstdint>
#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace vm {

class DTREagerBlobObject;
class DtrCudaAllocator final : public Allocator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DtrCudaAllocator);
  explicit DtrCudaAllocator(int64_t device_id);
  ~DtrCudaAllocator() override;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
  void Mark(DTREagerBlobObject* ebo, char* mem_ptr);
  void DisplayAllPieces();
  size_t allocated_memory();
  void set_left(bool is_left) { left = is_left; }
  bool left = true;

 private:
  using offset_t = size_t;

  offset_t get_offset(const char* mem_ptr) const;

  // Piece is the basic memory unit of CudaAllocator.
  // A Piece is either is free(is_free = true) or in used(is_free = false).
  // Pieces are stored in a linked list. The Piece's prev and next are
  // continuous with the current Piece in physical memory.
  struct Piece {
    size_t size = 0;
    char* ptr = nullptr;
    bool is_free = false;
    Piece* prev = nullptr;
    Piece* next = nullptr;
    vm::DTREagerBlobObject* tensor = nullptr;
    bool is_left = true;
  };

  bool InSmallMemoryArea(void* ptr);

  offset_t FindProperPositionInGroup(size_t group_idx, size_t request_size);

  Piece* AllocateMemoryInPiece(Piece* piece, offset_t offset_in_piece, size_t size);

  void InsertToFreeList(Piece* piece);
  void EraseFromFreeList(Piece* piece);

  // Try find free Piece which size is larger than aligned_size
  // Return nullptr when find failure
  Piece* FindPiece(size_t aligned_size, bool after_eviction);
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

  Piece* EvictAndFindPieceOnce(size_t size);
  Piece* EvictAndFindPieceMegEngineStyle(size_t size);

  int64_t device_id_;
  void* memory_ = nullptr;
  size_t memory_size_;
  void* small_piece_area_ptr_ = nullptr;

  // hold the lifetime of Piece
  std::vector<std::unique_ptr<Piece>> pieces_;
  struct PieceCmp {
    bool operator()(const Piece* lhs, const Piece* rhs) const {
      if (lhs->size != rhs->size) { return lhs->size < rhs->size; }
      return lhs->ptr < rhs->ptr;
    }
  };
  std::set<Piece*, PieceCmp> free_pieces_;
  // std::map is sorted by key, so we can find contiguous memory by it
  std::map<char*, Piece*> ptr2piece_;
  Piece* recycle_piece_list_;
  size_t total_allocate_bytes_ = 0;
  size_t total_deallocate_bytes_ = 0;

  // -----
  // size_t group_num_;
  // size_t cur_group_index_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_DTR_CUDA_ALLOCATOR_H_
