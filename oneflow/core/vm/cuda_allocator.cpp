#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

namespace {

inline size_t CudaMemAlignedBytes(size_t bytes) {
  return RoundUp(bytes, kCudaMemAllocAlignSize);
}

inline bool IsAlignedSize(size_t size) {
  return size % kCudaMemAllocAlignSize == 0;
}

static const size_t kPieceSplitThreshold = 128 << 20;  // 128MiB

} // namespace

CudaAllocator::CudaAllocator(int64_t device_id) : Allocator(), device_id_(device_id), recycle_piece_list_(nullptr) {
  cudaSetDevice(device_id_);
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  const size_t remain_bytes = 50 * 1048576;
  CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
  CHECK_GT(free_bytes, remain_bytes); // free bytes should greater than 50MiB
  size_t allocate_bytes = std::max(free_bytes - remain_bytes, static_cast<size_t>(free_bytes * 0.95));
  total_memory_bytes_ = CudaMemAlignedBytes(allocate_bytes);
  CHECK_LE(total_memory_bytes_, free_bytes);
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
  MarkPiece(piece);
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
  for(int32_t bin_num = BinNum4BinSize(aligned_size); bin_num < kBinNumSize; ++bin_num) {
    Bin* bin = &bins_.at(bin_num);
    for(auto it = bin->pieces.begin(); it != bin->pieces.end(); ++ it) {
      Piece* piece = *it;
      CHECK(piece->is_free);
      CHECK_NOTNULL(piece->ptr);
      CHECK_EQ(piece->bin_num, bin_num);
      CHECK_EQ(piece->size % kCudaMemAllocAlignSize, 0);
      if(piece->size >= aligned_size) {
        bin->pieces.erase(it);
        piece->bin_num = kInvalidBinNum;
        if(piece->size >= aligned_size * 2 || piece->size - aligned_size >= kPieceSplitThreshold) {
          Piece* new_piece = AllocatePiece();
          new_piece->ptr = piece->ptr + aligned_size;
          new_piece->size = piece->size - aligned_size;
          piece->size = aligned_size;

          Piece* next_p = piece->next;
          piece->next = new_piece;
          new_piece->prev = piece;
          new_piece->next = next_p;
          if(next_p != nullptr) {
            next_p->prev = new_piece;
          }
          
          new_piece->is_free = true;
          new_piece->bin_num = kInvalidBinNum;
          InsertPiece2Bin(new_piece);
        }
        return piece;
      }
    }
  }
  return nullptr;
}

void CudaAllocator::Allocate(char** mem_ptr, std::size_t size) {
  if(size == 0) { 
    *mem_ptr = nullptr;
    return; 
  }
  size_t aligned_size = CudaMemAlignedBytes(size);

  Piece* piece = FindPiece(aligned_size);
  CHECK(piece != nullptr) << "Error! : Out of memory when allocate size : " << size;
  CHECK_NOTNULL(piece->ptr);
  *mem_ptr = piece->ptr;
}

void CudaAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  TODO();
}

}  // namespace vm
}  // namespace oneflow
