#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

CudaAllocator::CudaAllocator(int64_t device_id) : Allocator(), device_id_(device_id), recycle_piece_list_(nullptr) {
  cudaSetDevice(device_id_);
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  const size_t remain_bytes = 10 * 1048576;
  CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
  CHECK_GT(free_bytes, remain_bytes); // free bytes should greater than 10MiB
  total_memory_bytes_ = std::max(free_bytes - remain_bytes, static_cast<size_t>(free_bytes * 0.95));
  CudaCheck(cudaMalloc(&mem_ptr_, total_memory_bytes_));

  bins_.resize(kBinNumSize);
  for(int i = 0; i < kBinNumSize; ++i) {
    size_t bin_size = BinSize4BinNum(i);
    bins_.at(i).size = bin_size;
    CHECK_EQ(BinNum4BinSize(bin_size), i);
    CHECK_EQ(BinNum4BinSize(bin_size + kCudaMemAllocAlignSize - 1), i);
    CHECK_EQ(BinNum4BinSize(bin_size * 2 - 1), i);
    CHECK_EQ(BinNum4BinSize(bin_size * 2), i == (kBinNumSize - 1) ? i : i + 1);
  }

  Piece* piece = AllocatePiece();
  piece->size = total_memory_bytes_;
  piece->ptr = mem_ptr_;
  piece->prev = nullptr;
  piece->next = nullptr;
  piece->is_free = true;
  piece->bin_num = kInvalidBinNum;

  InsertPiece2Bin(piece);
}

CudaAllocator::~CudaAllocator() {
  cudaSetDevice(device_id_);
  CudaCheck(cudaFree(mem_ptr_));
}

void CudaAllocator::InsertPiece2Bin(Piece* piece) {
  CHECK(piece->is_free && piece->bin_num == kInvalidBinNum);
  int32_t bin_num = BinNum4BinSize(piece->size);
  piece->bin_num = bin_num;
  CHECK(bins_.at(bin_num).pieces.insert(piece).second);
}

CudaAllocator::Piece* CudaAllocator::AllocatePiece() {
  if(recycle_piece_list_) {
    Piece* ret = recycle_piece_list_;
    recycle_piece_list_ = recycle_piece_list_->next;
    return ret;
  } else {
    pieces_.push_back(Piece());
    return &pieces_.at(pieces_.size() - 1);
  }
}

void CudaAllocator::Allocate(char** mem_ptr, std::size_t size) {
  TODO();
}

void CudaAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  TODO();
}

}  // namespace vm
}  // namespace oneflow
