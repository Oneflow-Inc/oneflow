#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

namespace {

inline size_t CudaMemAlignedBytes(size_t bytes) { return RoundUp(bytes, kCudaMemAllocAlignSize); }

inline bool IsAlignedSize(size_t size) { return size % kCudaMemAllocAlignSize == 0; }

static const size_t kPieceSplitThreshold = 128 << 20;  // 128MiB

}  // namespace

CudaAllocator::CudaAllocator(int64_t device_id)
    : Allocator(), device_id_(device_id), recycle_piece_list_(nullptr) {
  cudaSetDevice(device_id_);
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  const size_t remain_bytes = 50 * 1048576;
  CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
  CHECK_GT(free_bytes, remain_bytes);  // free bytes should greater than 50MiB
  size_t allocate_bytes =
      std::max(free_bytes - remain_bytes, static_cast<size_t>(free_bytes * 0.95));
  total_memory_bytes_ = CudaMemAlignedBytes(allocate_bytes);
  CHECK_LE(total_memory_bytes_, free_bytes);
  CudaCheck(cudaMalloc(&mem_ptr_, total_memory_bytes_));

  bins_.resize(kBinNumSize);
  for (int i = 0; i < kBinNumSize; ++i) {
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
  CHECK(IsAlignedSize(piece->size));
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
    /*
    for (int i = 0; i < pieces_.size(); ++i) {
      Piece* log_i = pieces_.at(i).get();
      LOG(INFO) << "cclog: allocate piece. old piece i = " << i << " piece->size = " << log_i->size
                << " piece address" << log_i;
    }
    */
    pieces_.emplace_back(new Piece());
    // pieces_.resize(pieces_.size() + 1);
    /*
    for (int i = 0; i < pieces_.size(); ++i) {
      Piece* log_i = pieces_.at(i).get();
      LOG(INFO) << "cclog: allocate piece. new piece i = " << i << " piece->size = " << log_i->size
                << " piece address" << log_i;
    }
    */
    // pieces_.push_back(Piece());
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
      // LOG(INFO) << "cclog: piece_size: " << piece->size;
      CHECK(IsAlignedSize(piece->size));
      if (piece->size >= aligned_size) {
        bin->pieces.erase(it);
        // LOG(INFO) << "cclog(1): piece_size: " << piece->size;
        piece->bin_num = kInvalidBinNum;
        piece->is_free = false;
        // LOG(INFO) << "cclog: Erase from Bin";
        if (piece->size >= aligned_size * 2 || piece->size - aligned_size >= kPieceSplitThreshold) {
          // LOG(INFO) << "cclog(2): piece_size: " << piece->size << " piece address" << piece;
          // LOG(INFO) << "cclog: Split piece START";
          /*
          for (int i = 0; i < pieces_.size(); ++i) {
            Piece* log_i = pieces_.at(i).get();
            LOG(INFO) << "cclog: before allocate piece. new piece i = " << i
                      << " piece->size = " << log_i->size << " piece address" << log_i;
          }
          */
          Piece* new_piece = AllocatePiece();
          /*
          for (int i = 0; i < pieces_.size(); ++i) {
            Piece* log_i = pieces_.at(i).get();
            LOG(INFO) << "cclog: after allocate piece. new piece i = " << i
                      << " piece->size = " << log_i->size << " piece address" << log_i;
          }
          LOG(INFO) << "cclog: Allocate new piece";
          LOG(INFO) << "cclog(3): piece_size: " << piece->size << " piece address" << piece;
          */
          new_piece->ptr = piece->ptr + aligned_size;
          /*
          LOG(INFO) << "cclog(4): piece_size: " << piece->size;
          LOG(INFO) << "cclog: new_piece_size = " << (piece->size - aligned_size)
                    << " piece_size = " << piece->size << " aligned_size = " << aligned_size;
          */
          new_piece->size = piece->size - aligned_size;
          piece->size = aligned_size;

          Piece* next_p = piece->next;
          piece->next = new_piece;
          new_piece->prev = piece;
          new_piece->next = next_p;
          if (next_p != nullptr) { next_p->prev = new_piece; }

          new_piece->is_free = true;
          new_piece->bin_num = kInvalidBinNum;
          // LOG(INFO) << "cclog: split piece_size: " << piece->size;
          // LOG(INFO) << "cclog: split new_piece_size: " << new_piece->size;
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

void CudaAllocator::Allocate(char** mem_ptr, std::size_t size) {
  if (size == 0) {
    *mem_ptr = nullptr;
    return;
  }
  size_t aligned_size = CudaMemAlignedBytes(size);

  Piece* piece = FindPiece(aligned_size);
  CHECK(piece != nullptr) << "Error! : Out of memory when allocate size : " << size;
  CHECK_NOTNULL(piece->ptr);
  CHECK(ptr2piece_.find(piece->ptr) != ptr2piece_.end());
  *mem_ptr = piece->ptr;
}

void CudaAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  if (mem_ptr == nullptr) { return; }

  auto it = ptr2piece_.find(mem_ptr);
  CHECK(it != ptr2piece_.end()) << "Error! : Try deallocate mem_ptr non-existent. mem ptr = "
                                << mem_ptr;
  Piece* piece = it->second;
  CHECK_NOTNULL(piece);
  CHECK_EQ(piece->ptr, mem_ptr);
  CHECK(!piece->is_free);

  piece->is_free = true;

  Piece* last_piece_insert_to_bin = piece;
  Piece* next_p = piece->next;
  Piece* prev_p = piece->prev;

  if (next_p != nullptr && next_p->is_free && next_p->ptr == piece->ptr + piece->size) {
    RemovePieceFromBin(next_p);
    MergeNeighbourFreePiece(piece, next_p);
  }

  if (prev_p != nullptr && prev_p->is_free && piece->ptr == prev_p->ptr + piece->size) {
    RemovePieceFromBin(prev_p);
    MergeNeighbourFreePiece(prev_p, piece);
    last_piece_insert_to_bin = prev_p;
  }
  InsertPiece2Bin(last_piece_insert_to_bin);
}

}  // namespace vm
}  // namespace oneflow
