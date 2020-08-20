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

#ifdef WITH_CUDA

#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include <iostream>

namespace oneflow {
namespace vm {

namespace {

inline size_t CudaMemAlignedBytes(size_t bytes) { return RoundUp(bytes, kCudaMemAllocAlignSize); }

inline bool IsAlignedSize(size_t size) { return size % kCudaMemAllocAlignSize == 0; }

static const size_t kPieceSplitThreshold = 128 << 20;  // 128MiB

}  // namespace

CudaAllocator::CudaAllocator(int64_t device_id)
    : Allocator(), device_id_(device_id), total_memory_bytes_(0), recycle_piece_list_(nullptr) {
  bins_.resize(kBinNumSize);
  for (int i = 0; i < kBinNumSize; ++i) {
    size_t bin_size = BinSize4BinNum(i);
    bins_.at(i).size = bin_size;
    CHECK_EQ(BinNum4BinSize(bin_size), i);
    CHECK_EQ(BinNum4BinSize(bin_size + kCudaMemAllocAlignSize - 1), i);
    CHECK_EQ(BinNum4BinSize(bin_size * 2 - 1), i);
    CHECK_EQ(BinNum4BinSize(bin_size * 2), i == (kBinNumSize - 1) ? i : i + 1);
  }
}

CudaAllocator::~CudaAllocator() {
  if (total_memory_bytes_ == 0) {
    CHECK_EQ(mem_ptr2block_.size(), 0);
    return;
  }
  cudaSetDevice(device_id_);
  for (auto& pair : mem_ptr2block_) { OF_CUDA_CHECK(cudaFree(pair.first)); }
}

void CudaAllocator::InsertPiece2Bin(Piece* piece) {
  CHECK(piece->is_free && piece->bin_num == kInvalidBinNum);
  int32_t bin_num = BinNum4BinSize(piece->size);
  piece->bin_num = bin_num;
  CHECK(bins_.at(bin_num).pieces.insert(piece).second);
}

void CudaAllocator::RemovePieceFromBin(Piece* piece) {
  CHECK(piece->is_free);
  CHECK_NE(piece->bin_num, kInvalidBinNum);
  CHECK_GT(bins_.at(piece->bin_num).pieces.erase(piece), 0);
  piece->bin_num = kInvalidBinNum;
}

CudaAllocator::Piece* CudaAllocator::AllocatePiece() {
  if (recycle_piece_list_) {
    Piece* ret = recycle_piece_list_;
    recycle_piece_list_ = recycle_piece_list_->next;
    return ret;
  } else {
    pieces_.emplace_back(new Piece());
    return pieces_.at(pieces_.size() - 1).get();
  }
}

void CudaAllocator::DeallocatePiece(Piece* piece) {
  piece->ptr = nullptr;
  piece->size = 0;
  piece->bin_num = kInvalidBinNum;
  piece->is_free = true;
  piece->prev = nullptr;
  piece->next = recycle_piece_list_;
  recycle_piece_list_ = piece;
}

void CudaAllocator::MarkPiece(Piece* piece) {
  CHECK_NOTNULL(piece->ptr);
  CHECK(ptr2piece_.emplace(piece->ptr, piece).second);
}
void CudaAllocator::UnMarkPiece(Piece* piece) {
  CHECK_NOTNULL(piece->ptr);
  auto it = ptr2piece_.find(piece->ptr);
  CHECK(it != ptr2piece_.end());
  ptr2piece_.erase(it);
}

CudaAllocator::Piece* CudaAllocator::FindPiece(size_t aligned_size) {
  CHECK(IsAlignedSize(aligned_size));
  for (int32_t bin_num = BinNum4BinSize(aligned_size); bin_num < kBinNumSize; ++bin_num) {
    Bin* bin = &bins_.at(bin_num);
    for (auto it = bin->pieces.begin(); it != bin->pieces.end(); ++it) {
      Piece* piece = *it;
      CHECK(piece->is_free);
      CHECK_NOTNULL(piece->ptr);
      CHECK_EQ(piece->bin_num, bin_num);
      CHECK(IsAlignedSize(piece->size));
      if (piece->size >= aligned_size) {
        bin->pieces.erase(it);
        piece->bin_num = kInvalidBinNum;
        piece->is_free = false;
        if (piece->size >= aligned_size * 2 || piece->size - aligned_size >= kPieceSplitThreshold) {
          Piece* new_piece = AllocatePiece();
          new_piece->ptr = piece->ptr + aligned_size;
          new_piece->size = piece->size - aligned_size;
          piece->size = aligned_size;

          Piece* next_p = piece->next;
          piece->next = new_piece;
          new_piece->prev = piece;
          new_piece->next = next_p;
          if (next_p != nullptr) { next_p->prev = new_piece; }

          new_piece->is_free = true;
          new_piece->bin_num = kInvalidBinNum;
          CHECK(IsAlignedSize(piece->size));
          CHECK(IsAlignedSize(new_piece->size));
          InsertPiece2Bin(new_piece);
          MarkPiece(new_piece);
        }
        return piece;
      }
    }
  }
  return nullptr;
}

void CudaAllocator::MergeNeighbourFreePiece(Piece* lhs, Piece* rhs) {
  CHECK(lhs->is_free);
  CHECK(rhs->is_free);
  CHECK(lhs->next == rhs);
  CHECK(lhs == rhs->prev);
  CHECK(lhs->ptr + lhs->size == rhs->ptr);

  lhs->size += rhs->size;
  lhs->next = rhs->next;
  if (rhs->next != nullptr) { rhs->next->prev = lhs; }
  UnMarkPiece(rhs);
  DeallocatePiece(rhs);
}

bool CudaAllocator::AllocateBlockToExtendTotalMem(size_t aligned_size) {
  CHECK(IsAlignedSize(aligned_size));

  size_t allocate_bytes = 1048576;  // 1MiB base size
  allocate_bytes = std::max(allocate_bytes, aligned_size);

  cudaSetDevice(device_id_);
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  OF_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t remain_bytes = 50 * 1048576;
  const size_t available_bytes = free_bytes - remain_bytes;  // remain at least 50MiB memory

  // growth double total memory bytes if could
  if (total_memory_bytes_ > 0) {
    allocate_bytes = std::max(allocate_bytes, std::min(total_memory_bytes_, available_bytes));
  }
  const size_t final_allocate_bytes = CudaMemAlignedBytes(allocate_bytes);

  if (final_allocate_bytes > available_bytes) { return false; }

  if (final_allocate_bytes < aligned_size) { return false; }

  char* mem_ptr = nullptr;
  if (cudaMalloc(&mem_ptr, final_allocate_bytes) != cudaSuccess) { return false; }

  // extend sucess
  total_memory_bytes_ += final_allocate_bytes;

  Piece* piece = AllocatePiece();
  piece->size = final_allocate_bytes;
  piece->ptr = mem_ptr;
  piece->prev = nullptr;
  piece->next = nullptr;
  piece->is_free = true;
  piece->bin_num = kInvalidBinNum;
  InsertPiece2Bin(piece);
  MarkPiece(piece);

  CHECK(mem_ptr2block_.emplace(mem_ptr, Block(piece)).second);

  return true;
}

bool CudaAllocator::DeallocateFreeBlockForGarbageCollection() {
  size_t total_free_bytes = 0;
  HashSet<char*> free_block_ptrs;
  for (const auto& pair : mem_ptr2block_) {
    const Block& block = pair.second;
    bool all_free = true;
    Piece* p = block.start_piece;
    while (p != nullptr) {
      if (!(p->is_free)) {
        all_free = false;
        break;
      }
      p = p->next;
    }

    if (all_free) {
      total_free_bytes += block.size;
      free_block_ptrs.insert(pair.first);
    }
  }

  total_memory_bytes_ -= total_free_bytes;

  if (total_free_bytes > 0) {
    LOG(WARNING) << "CudaAllocator try deallocate free block for garbage collection. "
                 << " deallocate free bytes : " << total_free_bytes;
    cudaSetDevice(device_id_);
    for (char* ptr : free_block_ptrs) {
      auto it = mem_ptr2block_.find(ptr);
      CHECK(it != mem_ptr2block_.end());
      const Block& block = it->second;

      // delete all Piece on Block
      size_t piece_size_sum = 0;
      Piece* p = block.start_piece;
      CHECK_EQ(block.ptr, block.start_piece->ptr);
      CHECK_EQ(block.ptr, ptr);
      while (p != nullptr) {
        Piece* next_p = p->next;
        piece_size_sum += p->size;
        RemovePieceFromBin(p);
        UnMarkPiece(p);
        DeallocatePiece(p);
        p = next_p;
      }
      CHECK_EQ(block.size, piece_size_sum);

      mem_ptr2block_.erase(it);
      OF_CUDA_CHECK(cudaFree(ptr));
    }
  }

  return total_free_bytes > 0;
}

void CudaAllocator::Allocate(char** mem_ptr, std::size_t size) {
  if (size == 0) {
    *mem_ptr = nullptr;
    return;
  }
  size_t aligned_size = CudaMemAlignedBytes(size);

  Piece* piece = FindPiece(aligned_size);
  if (piece == nullptr) {
    if (AllocateBlockToExtendTotalMem(aligned_size)) { piece = FindPiece(aligned_size); }
  }

  if (piece == nullptr) {
    if (DeallocateFreeBlockForGarbageCollection() && AllocateBlockToExtendTotalMem(aligned_size)) {
      piece = FindPiece(aligned_size);
    }
  }

  CHECK(piece != nullptr) << "Error! : Out of memory when allocate size : " << size;
  CHECK_NOTNULL(piece->ptr);
  CHECK(ptr2piece_.find(piece->ptr) != ptr2piece_.end());
  *mem_ptr = piece->ptr;
}

void CudaAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  if (mem_ptr == nullptr) { return; }

  auto it = ptr2piece_.find(mem_ptr);
  CHECK(it != ptr2piece_.end()) << "Error! : Try deallocate mem_ptr non-existent. mem ptr = "
                                << mem_ptr << " size = " << size;
  Piece* piece = it->second;
  CHECK_NOTNULL(piece);
  CHECK_EQ(piece->ptr, mem_ptr);
  CHECK(!piece->is_free);

  piece->is_free = true;

  Piece* last_piece_insert_to_bin = piece;
  Piece* next_p = piece->next;
  Piece* prev_p = piece->prev;

  if (next_p != nullptr && next_p->is_free) {
    CHECK_EQ(next_p->ptr, piece->ptr + piece->size);
    RemovePieceFromBin(next_p);
    MergeNeighbourFreePiece(piece, next_p);
  }

  if (prev_p != nullptr && prev_p->is_free) {
    CHECK_EQ(piece->ptr, prev_p->ptr + prev_p->size);
    RemovePieceFromBin(prev_p);
    MergeNeighbourFreePiece(prev_p, piece);
    last_piece_insert_to_bin = prev_p;
  }
  InsertPiece2Bin(last_piece_insert_to_bin);
}

}  // namespace vm
}  // namespace oneflow

#endif
