#ifndef ONEFLOW_CORE_VM_CUDA_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_CUDA_ALLOCATOR_H_

#include <cstdint>
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

class CudaAllocator final : public Allocator {
 public:
  explicit CudaAllocator(int64_t device_id);
  ~CudaAllocator() override;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;

 private:
  static constexpr int32_t kInvalidBinNum = -1;
  static constexpr int32_t kBinNumSize = 20;

  struct Piece {
    size_t size = 0;
    char* ptr = nullptr;
    bool is_free = false;
    Piece* prev = nullptr;
    Piece* next = nullptr;
    int32_t bin_num = kInvalidBinNum;
  };

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

  size_t BinSize4BinNum(int32_t bin_num) { return kCudaMemAllocAlignSize << bin_num; }

  int32_t BinNum4BinSize(size_t size) {
    uint64_t value = std::max(size, kCudaMemAllocAlignSize) >> 9;
    return std::min(kBinNumSize - 1, static_cast<int32_t>(63 ^ __builtin_clzll(value)));
  }

  Piece* FindPiece(size_t aligned_size);
  void InsertPiece2Bin(Piece* piece);
  Piece* AllocatePiece();
  void DeallocatePiece(Piece* piece);
  void MarkPiece(Piece* piece);
  void UnMarkPiece(Piece* piece);
  void MergeNeighbourFreePiece(Piece* lhs, Piece* rhs);
  void RemovePieceFromBin(Piece* piece);

  int64_t device_id_;
  size_t total_memory_bytes_;
  char* mem_ptr_;  // maybe ptr list for dynamic growth

  std::vector<Bin> bins_;
  std::vector<std::unique_ptr<Piece>> pieces_;
  HashMap<char*, Piece*> ptr2piece_;
  Piece* recycle_piece_list_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_ALLOCATOR_H_
