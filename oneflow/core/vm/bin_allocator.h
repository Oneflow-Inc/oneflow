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
#ifndef ONEFLOW_CORE_VM_BIN_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_BIN_ALLOCATOR_H_

#include <cstdint>
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/shrinkable_cache.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

class BinAllocator final : public Allocator, public ShrinkableCache {
 public:
  explicit BinAllocator(size_t alignment, std::unique_ptr<Allocator>&& backend);
  ~BinAllocator() override;

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
  void Shrink() override { DeallocateFreeBlockForGarbageCollection(); }
  void DeviceReset() override { backend_->DeviceReset(); }

 private:
  static constexpr int32_t kInvalidBinNum = -1;
  static constexpr int32_t kBinNumSize = 20;

  // Piece is the basic memory unit of BinAllocator.
  // A Piece is either is free(is_free = true) or in used(is_free = false).
  // If the Piece is_free = true, the pointer to the piece will be stored in the Bin structure of
  // the corresponding BinSize. Pieces are stored in a linked list. The Piece's prev and next are
  // continuous with the current Piece in physical memory.
  struct Piece {
    size_t size = 0;
    char* ptr = nullptr;
    bool is_free = false;
    Piece* prev = nullptr;
    Piece* next = nullptr;
    int32_t bin_num = kInvalidBinNum;
  };

  // Bin is a structure that stores a set of pieces which is free and has similar size, and
  // these Pieces are arger than the size of bin
  //
  // BinAllocator has a set of Bin structures according to the binary multiple increasing relation,
  // which is used to quickly index and find the free Piece of appropriate size when Allocate()
  //
  // The size of the smallest bin is 512 (512 is the smallest unit Allocated by BinAllocator,
  // and the memory size of all Allocated will be multiples of 512, 512 is kCudaMemAllocAlignSize).
  // The size of each Bin is twice the size of the previous Bin, like
  //    BinNum:   Bin0, Bin1, Bin2, Bin3, ..., Bin19
  //    BinSize:  512, 1024, 2048, 4096, ... , 512MB
  struct Bin {
    size_t size = 0;

    struct PieceCmp {
      bool operator()(const Piece* lhs, const Piece* rhs) const {
        if (lhs->size != rhs->size) { return lhs->size < rhs->size; }
        return lhs->ptr < rhs->ptr;
      }
    };
    std::set<Piece*, PieceCmp> pieces;
  };

  // Block is large physical memory that is actually allocated.
  // There maybe many consecutive disjoint Pieces distributed on the Block memory
  struct Block {
    size_t size = 0;
    char* ptr = nullptr;
    Piece* start_piece = nullptr;
    Block(Piece* p) : size(p->size), ptr(p->ptr), start_piece(p) {}
  };

  size_t BinSize4BinNum(int32_t bin_num) { return kCudaMemAllocAlignSize << bin_num; }

  int32_t BinNum4BinSize(size_t size) {
    uint64_t value = std::max(size, kCudaMemAllocAlignSize) >> 9;
    return std::min(kBinNumSize - 1, static_cast<int32_t>(63 ^ __builtin_clzll(value)));
  }

  // Try find free Piece which size is larger than aligned_size in Bins.
  // Return nullptr when find failure
  Piece* FindPiece(size_t aligned_size);

  // Insert the free Piece to the appropriate Bin which bin size is smaller than piece
  void InsertPiece2Bin(Piece* piece);

  // Create new empty Piece or recycle a Piece from recycle_piece_list_
  Piece* AllocatePiece();
  // Delete a Piece and move in the linked list recycle_piece_list_
  void DeallocatePiece(Piece* piece);

  // Insert a {piece->ptr, piece} pair into the ptr2piece_ map for search Piece when call
  // Deallocate()
  void MarkPiece(Piece* piece);
  // Erase the {piece->ptr, piece} pair from ptr2piece_ because the ptr is useless
  // Usually call before DeallocatePiece()
  void UnMarkPiece(Piece* piece);

  void MergeNeighbourFreePiece(Piece* lhs, Piece* rhs);
  void RemovePieceFromBin(Piece* piece);

  Maybe<bool> AllocateBlockToExtendTotalMem(size_t aligned_size);
  bool DeallocateFreeBlockForGarbageCollection();

  const size_t alignment_;
  const std::unique_ptr<Allocator> backend_;
  size_t total_memory_bytes_;
  HashMap<char*, Block> mem_ptr2block_;

  std::vector<Bin> bins_;
  std::vector<std::unique_ptr<Piece>> pieces_;
  HashMap<char*, Piece*> ptr2piece_;
  Piece* recycle_piece_list_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_BIN_ALLOCATOR_H_
