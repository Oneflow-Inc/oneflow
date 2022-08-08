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

#include <iterator>
#include <vector>
#include "nlohmann/json.hpp"
#include "oneflow/core/profiler/util.h"
#ifdef WITH_CUDA

#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include <iostream>

namespace oneflow {
namespace vm {

namespace {

inline size_t CudaMemAlignedBytes(size_t bytes) { return RoundUp(bytes, kCudaMemAllocAlignSize); }

inline bool IsAlignedSize(size_t size) { return size % kCudaMemAllocAlignSize == 0; }

inline double bytes2Mb(size_t bytes) { return bytes * 1. / 1024 / 1024; }

static constexpr size_t kSmallPieceThreshold = 10 * 1024;  // 10 KB

inline bool ShouldBeHeldBySmallPiece(size_t size) {
  return EnvBool<ONEFLOW_DTR_SMALL_PIECE>() && size <= kSmallPieceThreshold;
}

std::vector<size_t> GroupNumToIndexes(size_t group_num) {
  switch (group_num) {
    case 1: return {0};
    case 2: return {0, 1};
    case 3: return {0, 1, 2};
    case 4: return {0, 1, 3, 2};
    case 6: return {3, 1, 0, 5, 4, 2};
  }
  UNIMPLEMENTED();
}

}  // namespace

DtrCudaAllocator::DtrCudaAllocator(int64_t device_id)
    : Allocator(),
      device_id_(device_id),
      memory_size_(0),
      recycle_piece_list_(nullptr),
      normal_group_num_(EnvInteger<ONEFLOW_DTR_GROUP_NUM>()),
      group_indexes_(GroupNumToIndexes(normal_group_num_)),
      cur_group_index_id_(EnvBool<OF_DTR_NLR>() && normal_group_num_ > 1 ? 1 : 0),
      cur_group_index_id_high_cost_(0),
      enable_left_and_right_(EnvBool<OF_DTR_LR>() && normal_group_num_ > 1) {
  free_pieces_overlapping_with_group_.resize(normal_group_num_ + 1);
}

DtrCudaAllocator::~DtrCudaAllocator() {
  cudaSetDevice(device_id_);
  if (memory_ != nullptr) { OF_CUDA_CHECK(cudaFree(memory_)); }
}

DtrCudaAllocator::offset_t DtrCudaAllocator::get_offset(const char* mem_ptr) const {
  return mem_ptr - (char*)memory_;
}

void DtrCudaAllocator::Mark(DTREagerBlobObject* ebo, char* mem_ptr) {
  Piece* piece = ptr2piece_.at(mem_ptr);
  piece->tensor = ebo;
  if (dtr::debug_level() >= 1) {
    LOG(INFO) << "tensor " << ebo->id() << " is allocated at " << get_offset(mem_ptr)
              << ", left: " << piece->is_left;
  }
}

bool DtrCudaAllocator::InSmallMemoryArea(void* ptr) {
  CHECK_NOTNULL(small_piece_area_ptr_);
  CHECK_GE(ptr, memory_);
  CHECK_LT(ptr, (char*)memory_ + memory_size_);
  // compare pointer by raw < or > is undefined behavior
  return std::greater_equal<>{}(ptr, small_piece_area_ptr_);
}

DtrCudaAllocator::Piece* DtrCudaAllocator::AllocatePiece() {
  if (recycle_piece_list_) {
    Piece* ret = recycle_piece_list_;
    recycle_piece_list_ = recycle_piece_list_->next;
    return ret;
  } else {
    pieces_.emplace_back(new Piece());
    return pieces_.at(pieces_.size() - 1).get();
  }
}

void DtrCudaAllocator::DeallocatePiece(Piece* piece) {
  piece->ptr = nullptr;
  piece->size = 0;
  CHECK(piece->is_free);
  piece->prev = nullptr;
  piece->next = recycle_piece_list_;
  piece->is_left = true;
  recycle_piece_list_ = piece;
}

void DtrCudaAllocator::InsertPiece2PtrMap(Piece* piece) {
  CHECK_NOTNULL(piece->ptr);
  CHECK(ptr2piece_.emplace(piece->ptr, piece).second);
}

void DtrCudaAllocator::ErasePieceFromPtrMap(Piece* piece) {
  CHECK_NOTNULL(piece->ptr);
  auto it = ptr2piece_.find(piece->ptr);
  CHECK(it != ptr2piece_.end());
  ptr2piece_.erase(it);
}

double get_cost(const vm::DTREagerBlobObject* ebo, int& coeff, size_t size) {
  if (ebo == nullptr) { return 0.; }
  double cost = CHECK_JUST(ebo->cost(size));

  if (!EnvBool<OF_DTR_O_ONE>()) {
    // CHECK(!isinf(cost));
    CHECK(!isnan(cost));
    return cost;
  }
  // const double cost = CHECK_JUST(ebo->cost());
  if (coeff < 0) { coeff = ebo->pesudo_cnt(); }
  cost = cost * coeff;
  // CHECK(!isinf(cost));
  CHECK(!isnan(cost));
  return cost;
}

double get_cost(const vm::DTREagerBlobObject* ebo, int& coeff) { return get_cost(ebo, coeff, 0); }

void DtrCudaAllocator::DisplayAllPieces() {
  for (const auto& pair : ptr2piece_) {
    Piece* piece = pair.second;
    std::stringstream ss;
    ss << "piece " << piece << ", " << (void*)piece->ptr << ", " << piece->size << ", ";
    int coeff = -1;
    if (piece->tensor) {
      ss << "ebo: " << piece->tensor << ", id: " << piece->tensor->id()
         << ", cost: " << get_cost(piece->tensor, coeff)
         << ", pinned: " << piece->tensor->num_pinned()
         << ", evictable: " << piece->tensor->is_evictable()
         << ", compute op: " << piece->tensor->compute_op_type_name();
    } else {
      ss << "no tensor";
    }
    std::cout << ss.str() << std::endl;
  }
}

void DtrCudaAllocator::Display() {
  double total_free_piece_bytes = 0.;
  for (const auto& free_list : free_pieces_overlapping_with_group_) {
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
      Piece* piece = *it;
      CHECK(piece->is_free);
      CHECK_NOTNULL(piece->ptr);
      CHECK(IsAlignedSize(piece->size));
      std::cout << "memory: " << piece->size * 1. / 1024 / 1024 << "MB" << std::endl;
      total_free_piece_bytes += piece->size;
    }
  }
  std::cout << "total_free_piece_bytes: " << bytes2Mb(total_free_piece_bytes) << "MB"
            << ", total allocate bytes: " << bytes2Mb(total_allocate_bytes_) << "MB"
            << ", total deallocate bytes: " << bytes2Mb(total_deallocate_bytes_) << "MB"
            << std::endl;
}

// 开启了 left-right 之后，才能开启 op guided
DtrCudaAllocator::offset_t DtrCudaAllocator::FindProperPositionInGroup(Piece* piece,
                                                                       size_t group_idx,
                                                                       size_t request_size) const {
  const offset_t grp_left_bound = group_boundaries_[group_idx].first;
  const offset_t grp_right_bound = group_boundaries_[group_idx].second;
  const offset_t piece_left_bound = get_offset(piece->ptr);
  const offset_t piece_right_bound = piece_left_bound + piece->size;
  const bool is_right =
      enable_left_and_right_ && (group_idx % 2 == 1) && group_idx != normal_group_num_;
#define PNT(var) LOG(INFO) << OF_PP_STRINGIZE(var) << ": " << var << std::endl
  PNT(group_idx);
  PNT(grp_left_bound);
  PNT(grp_right_bound);
  PNT(piece_left_bound);
  PNT(piece_right_bound);
  PNT(is_right);
  PNT(request_size);

  if (is_right) {
    if (grp_right_bound < piece_right_bound) {
      if (grp_right_bound - request_size > piece_left_bound) {
        return grp_right_bound - request_size;
      }
    }
    // half of tensor in group
    if (piece_right_bound - request_size / 2 < grp_right_bound) {
      return piece_right_bound - request_size;
    }
  } else {
    if (grp_left_bound > piece_left_bound) {
      if (grp_left_bound + request_size < piece_right_bound) { return grp_left_bound; }
    }
    // half of tensor in group
    if (piece_left_bound + request_size / 2 > grp_left_bound) { return piece_left_bound; }
  }
  return SIZE_MAX;
}

void DtrCudaAllocator::InsertToFreeList(Piece* piece) {
  const offset_t piece_left = get_offset(piece->ptr);
  const offset_t piece_right = piece_left + piece->size;
  LOG(INFO) << "piece_left: " << piece_left << ", right: " << piece_right << std::endl;
  for (size_t i = 0; i < group_boundaries_.size(); i++) {
    LOG(INFO) << "g left: " << group_boundaries_[i].first
              << ", right: " << group_boundaries_[i].second << std::endl;
    if ((piece_left >= group_boundaries_[i].first && piece_left < group_boundaries_[i].second)
        || (piece_right > group_boundaries_[i].first
            && piece_right <= group_boundaries_[i].second)) {
      LOG(INFO) << "overlap" << std::endl;
      free_pieces_overlapping_with_group_[i].insert(piece);
    }
  }
}

void DtrCudaAllocator::EraseFromFreeList(Piece* piece) {
  LOG(INFO) << "erase " << get_offset(piece->ptr);
  // NOTE: very strange bug:
  // std::map::erase(Key) returns 2 instead of 0 or 1, which conflicts with documentation.
  for (auto& free_list : free_pieces_overlapping_with_group_) {
    for (auto it = free_list.begin(); it != free_list.end(); it++) {
      if ((*it)->ptr == piece->ptr) {
        free_list.erase(it);
        break;
      }
    }
  }
}

auto DtrCudaAllocator::AllocateMemoryInPiece(Piece* piece, offset_t offset_in_piece, size_t size)
    -> Piece* {
  auto SplitPiece = [this](Piece* piece, offset_t offset_in_piece) -> Piece* {
    // offset_in_piece must be less (not equal) than piece->size so that
    // new_piece has size
    CHECK_LE(offset_in_piece, piece->size);
    Piece* new_piece = AllocatePiece();
    new_piece->ptr = piece->ptr + offset_in_piece;
    new_piece->size = piece->size - offset_in_piece;
    piece->size = offset_in_piece;

    Piece* next_p = piece->next;
    piece->next = new_piece;
    new_piece->prev = piece;
    new_piece->next = next_p;
    if (next_p != nullptr) { next_p->prev = new_piece; }
    InsertPiece2PtrMap(new_piece);

    CHECK(IsAlignedSize(piece->size));
    CHECK(IsAlignedSize(new_piece->size));
    return new_piece;
  };
  auto SplitPiece3 = [&SplitPiece](
                         Piece* piece, offset_t offset1_in_piece,
                         offset_t offset2_in_piece) -> std::tuple<Piece*, Piece*, Piece*> {
    Piece* piece1 = nullptr;
    Piece* piece2 = nullptr;
    Piece* piece3 = nullptr;
    bool has_piece3 = offset2_in_piece != piece->size;
    if (offset1_in_piece > 0) {
      piece1 = piece;
      piece2 = SplitPiece(piece, offset1_in_piece);
    } else {
      piece1 = nullptr;
      piece2 = piece;
    }
    if (has_piece3) { piece3 = SplitPiece(piece2, offset2_in_piece - offset1_in_piece); }
    return {piece1, piece2, piece3};
  };
  auto pieces = SplitPiece3(piece, offset_in_piece, offset_in_piece + size);
  EraseFromFreeList(piece);
  Piece *piece1 = std::get<0>(pieces), *piece2 = std::get<1>(pieces), *piece3 = std::get<2>(pieces);
  if (piece1 != nullptr) {
    // piece1 is already free
    InsertToFreeList(piece1);
  }
  piece2->is_free = false;
  if (piece3 != nullptr) {
    piece3->is_free = true;
    InsertToFreeList(piece3);
  }
  return piece2;
}

size_t DtrCudaAllocator::iterate_group_index(bool high) const {
  auto is_high_group = [](size_t idx) -> bool { return (idx / 2) % 2 == (idx % 2); };
  if (high) {
    size_t index; // NOLINT
    do {
      cur_group_index_id_high_cost_ = (cur_group_index_id_high_cost_ + 1) % normal_group_num_;
      index = group_indexes_[cur_group_index_id_high_cost_];
    } while (!is_high_group(index));
    return index;
  } else {
    size_t index; // NOLINT
    do {
      cur_group_index_id_ = (cur_group_index_id_ + 1) % normal_group_num_;
      index = group_indexes_[cur_group_index_id_];
    } while (EnvBool<OF_DTR_NLR>() && is_high_group(index));
    return index;
  }
}

size_t DtrCudaAllocator::group_index(bool high) const {
  if (high) {
    return group_indexes_[cur_group_index_id_high_cost_];
  } else {
    return group_indexes_[cur_group_index_id_];
  }
}

void DtrCudaAllocator::InitMemory() {
  memory_size_ = dtr::memory_threshold();
  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "BEGINNING-" << memory_size_ << std::endl;
  }
  OF_CUDA_CHECK(cudaMalloc(&memory_, memory_size_));
  const size_t small_piece_area_size =
      EnvBool<ONEFLOW_DTR_SMALL_PIECE>() ? 1024 * kSmallPieceThreshold : 0;
  const size_t normal_area_size = memory_size_ - small_piece_area_size;
  small_piece_area_ptr_ = static_cast<char*>(memory_) + normal_area_size;

  if (enable_left_and_right_) { CHECK_EQ(normal_group_num_ % 2, 0); }
  const size_t effective_normal_group_num =
      enable_left_and_right_ ? normal_group_num_ / 2 : normal_group_num_;
  const std::vector<offset_t> boundary_tmp = [&]() {
    const size_t mem_per_group = normal_area_size / effective_normal_group_num;
    std::vector<offset_t> boundary_tmp;
    for (size_t i = 0, b = 0; i < effective_normal_group_num; i++, b += mem_per_group) {
      boundary_tmp.push_back(b);
    }
    boundary_tmp.push_back(normal_area_size);
    return boundary_tmp;
  }();
  for (size_t i = 0; i < effective_normal_group_num; i++) {
    group_boundaries_.emplace_back(boundary_tmp[i], boundary_tmp[i + 1]);
    if (enable_left_and_right_) {
      group_boundaries_.emplace_back(boundary_tmp[i], boundary_tmp[i + 1]);
    }
  }
  if (normal_area_size != memory_size_) {
    group_boundaries_.emplace_back(normal_area_size, memory_size_);
  }

  Piece* piece = AllocatePiece();
  piece->size = memory_size_;
  piece->ptr = static_cast<char*>(memory_);
  piece->prev = nullptr;
  piece->next = nullptr;
  piece->is_free = true;
  piece->tensor = nullptr;
  InsertToFreeList(piece);
  InsertPiece2PtrMap(piece);
}

DtrCudaAllocator::Piece* DtrCudaAllocator::FindPiece(size_t aligned_size, bool after_eviction) {
  CHECK(IsAlignedSize(aligned_size));

  if (memory_ == nullptr) { InitMemory(); }

  const bool is_high_op = [&]() {
    if (!EnvBool<OF_DTR_NLR>()) { return false; }
    std::vector<std::string> high_compute_cost_names{"conv2d", "conv_data_grad", "conv_filter_grad",
                                                     "add_n",  "matmul",         "batch_matmul"};
    const std::string& name = Global<dtr::TensorPool>::Get()->current_op_type_name();
    PNT(name);
    if (std::find(high_compute_cost_names.cbegin(), high_compute_cost_names.cend(), name)
        != high_compute_cost_names.cend()) {
      return true;
    }
    return false;
  }();

  size_t group_idx = [&]() -> size_t {
    if (ShouldBeHeldBySmallPiece(aligned_size)) { return normal_group_num_; }
    // if (after_eviction) { return true; }
    return group_index(is_high_op);
  }();
  PNT(aligned_size);
  size_t iterate_num = 0;
  do {
    const auto& free_pieces = free_pieces_overlapping_with_group_[group_idx];
    PNT(group_idx);
    PNT(free_pieces.size());
    for (auto it = free_pieces.begin(); it != free_pieces.end(); ++it) {
      Piece* piece = *it;
      CHECK(piece->is_free);
      CHECK_NOTNULL(piece->ptr);
      CHECK(IsAlignedSize(piece->size));
      PNT(get_offset(piece->ptr));
      PNT(piece->size);
      if (piece->size >= aligned_size) {
        const offset_t offset_in_memory = FindProperPositionInGroup(piece, group_idx, aligned_size);
        PNT(offset_in_memory);
        if (offset_in_memory != SIZE_MAX) {
          const offset_t offset_in_piece = offset_in_memory - get_offset(piece->ptr);
          return AllocateMemoryInPiece(piece, offset_in_piece, aligned_size);
        }
      }
    }
    // update group_idx only if this group fails
    // multiple outputs of a single op places in the same group
    group_idx = iterate_group_index(is_high_op);
    iterate_num++;
  } while (!ShouldBeHeldBySmallPiece(aligned_size) && iterate_num < normal_group_num_);

  return nullptr;
}

void DtrCudaAllocator::MergeNeighbourFreePiece(Piece* lhs, Piece* rhs) {
  CHECK(lhs->is_free);
  CHECK(rhs->is_free);
  CHECK(lhs->next == rhs);
  CHECK(lhs == rhs->prev);
  CHECK(lhs->ptr + lhs->size == rhs->ptr);

  lhs->size += rhs->size;
  lhs->next = rhs->next;
  if (rhs->next != nullptr) { rhs->next->prev = lhs; }
  ErasePieceFromPtrMap(rhs);
  DeallocatePiece(rhs);
}

DtrCudaAllocator::Piece* DtrCudaAllocator::EvictAndFindPieceMegEngineStyle(size_t required_size) {
  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "START-EvictAndFindPiece" << std::endl;
  }
  if (dtr::debug_level() >= 2) { LOG(INFO) << "required size: " << required_size; }
  auto GetSizeIncludingNeighborhood = [](auto it, auto begin, auto end) -> size_t {
    size_t size = it->second->size;
    if (it != begin) {
      for (auto t = std::prev(it); t->second->tensor == nullptr; t--) {
        size += t->second->size;
        if (t == begin) { break; }
      }
    }
    if (it != end) {
      for (auto t = std::next(it); t != end && t->second->tensor == nullptr; t++) {
        size += t->second->size;
      }
    }
    return size;
  };

  int i = 0;
  while (true) {
    i++;
    double min_cost = std::numeric_limits<double>::max();
    vm::DTREagerBlobObject* min_tensor = nullptr;
    for (auto it = ptr2piece_.begin();
         it != ptr2piece_.end() && !InSmallMemoryArea(it->second->ptr); it++) {
      int coeff = -1;
      auto* tensor = it->second->tensor;
      if (tensor != nullptr && !tensor->is_pinned() && tensor->is_evictable()) {
        auto cur_op_cost = get_cost(
            tensor, coeff, GetSizeIncludingNeighborhood(it, ptr2piece_.begin(), ptr2piece_.end()));
        if (cur_op_cost < min_cost) {
          min_cost = cur_op_cost;
          min_tensor = tensor;
        }
      }
    }
    if (min_tensor) {
      CHECK_JUST(min_tensor->evict(false));
      Piece* piece = FindPiece(required_size, true);
      if (piece != nullptr) { return piece; }
    } else {
      LOG(FATAL) << "eviction fail";
    }
  }
}

DtrCudaAllocator::Piece* DtrCudaAllocator::EvictAndFindPieceOnce(size_t required_size) {
  auto start = ptr2piece_.begin();
  auto end = ptr2piece_.begin();
  size_t total_size = 0;
  double cost_except_size = 0;
  double min_cost = std::numeric_limits<double>::max();
  auto min_start = start;
  auto min_end = start;
  std::vector<double> costs;
  costs.reserve(ptr2piece_.size());
  size_t start_i = 0;
  size_t end_i = 0;
  while (end != ptr2piece_.end() && !InSmallMemoryArea(end->second->ptr)) {
    if (total_size < required_size) {
      auto* end_tensor = end->second->tensor;
      if (end_tensor != nullptr && (end_tensor->is_pinned() || !end_tensor->is_evictable())) {
        end++;
        costs.push_back(0);
        end_i++;
        start = end;
        start_i = end_i;
        total_size = 0;
        cost_except_size = 0;
        continue;
      }
      total_size += end->second->size;
      // end_tensor is fakely evicted, update_after_pesudo_evict
      if (EnvBool<OF_DTR_O_ONE>() && end_tensor != nullptr) {
        const char* start_id = start->first;
        const char* end_id = end->first;
        CHECK_JUST(Global<dtr::TensorPool>::Get()->update_after_pesudo_evict(end_tensor, start_id,
                                                                             end_id));
      }
      int coeff = -1;
      auto cur_op_cost = get_cost(end_tensor, coeff);
      costs.push_back(cur_op_cost);
      cost_except_size += cur_op_cost;
      if (dtr::debug_level() >= 2) {
        LOG(INFO) << "move end, include op: "
                  << (end_tensor != nullptr ? end_tensor->compute_op_type_name() : "no tensor")
                  << ", size: " << end->second->size << ", total_size: " << total_size
                  << ", total cost: " << cost_except_size << ", cur op cost: " << cur_op_cost;
      }
      end++;
      end_i++;
    } else {
      auto* start_tensor = start->second->tensor;
      // const auto* start_tensor = start->second->tensor;
      total_size -= start->second->size;
      // start_tensor is back in the pool, update_after_pesudo_compute
      int coeff = -1;
      double cur_op_cost = 0;
      if (EnvBool<OF_DTR_O_ONE>() && start_tensor != nullptr) {
        coeff = Global<dtr::TensorPool>::Get()->update_after_pesudo_compute(start_tensor);
        cur_op_cost = get_cost(start_tensor, coeff);
      } else {
        cur_op_cost = costs[start_i];
      }
      cost_except_size -= cur_op_cost;
            if (dtr::debug_level() >= 2) {
        LOG(INFO) << "move start, exclude op: "
                  << (start_tensor != nullptr ? start_tensor->compute_op_type_name() : "no tensor")
                  << ", size: " << start->second->size << ", total_size: " << total_size
                  << ", total cost: " << cost_except_size << ", cur op cost: " << cur_op_cost;
      }
      start++;
      start_i++;
    }
    double cost = cost_except_size;
    if (EnvBool<ONEFLOW_DTR_HEURISTIC_WITH_SIZE>()) { cost /= total_size; }
    if (total_size >= required_size && cost < min_cost) {
      min_cost = cost;
      min_start = start;
      min_end = end;
      if (dtr::debug_level() >= 2) { LOG(INFO) << "record, min_cost: " << min_cost; }
    }
  }
  // CHECK(min_end != start);
  // collect piece ptrs into a new container, because evict() will devalidate the iterators
  std::vector<Piece*> pieces_to_be_evicted;
  for (auto it = min_start; it != min_end; ++it) {
    Piece* piece = it->second;
    pieces_to_be_evicted.push_back(piece);
  }
  if (dtr::is_enabled_and_debug()) {
    for (auto* piece : pieces_to_be_evicted) {
      int coeff = -1;
      LOG(INFO) << "release dptr: " << get_offset(piece->ptr) << ", size: " << piece->size
                << ", cost: " << get_cost(piece->tensor, coeff) << ", compute op: "
                << (piece->tensor != nullptr ? piece->tensor->compute_op_type_name() : "no")
                << ", id: "
                << (piece->tensor != nullptr ? std::to_string(piece->tensor->id()) : "no");
    }
  }
  for (auto* piece : pieces_to_be_evicted) {
    // NOTE: evict will trigger the merge and deallocation of neighbour free pieces,
    // e.g. two contiguous pieces relu, no_tensor, after relu evict, no_tensor will be deallocated.
    // currently deallocation only set tensor to nullptr, not real free,
    // so no bug occurs. It is tricky and fragile.
    if (piece->tensor != nullptr) {
      CHECK(!ShouldBeHeldBySmallPiece(piece->size));
      CHECK_JUST(piece->tensor->evict(false));
    }
  }

  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "END-EvictAndFindPiece" << std::endl;
  }

  if (!pieces_to_be_evicted.empty()) { return CHECK_NOTNULL(FindPiece(required_size, true)); }
  return nullptr;
}

bool first_time = true;

void DtrCudaAllocator::Allocate(char** mem_ptr, std::size_t size) {
  if (size == 0) {
    *mem_ptr = nullptr;
    return;
  }
  size_t aligned_size = CudaMemAlignedBytes(size);

  Piece* piece = FindPiece(aligned_size, false);

  if (piece == nullptr) {
    if (first_time) {
      if (EnvBool<ONEFLOW_DTR_DISPLAY_IN_FIRST_TIME>()) { DisplayAllPieces(); }
      first_time = false;
    }
    const auto started_at = profiler::GetTimeNow();
    const int evict_num1 = Global<dtr::TensorPool>::Get()->num_forced_eviction();
    if (EnvBool<ONEFLOW_DTR_MEGENGINE_STYLE>()) {
      piece = EvictAndFindPieceMegEngineStyle(aligned_size);
    } else {
      piece = EvictAndFindPieceOnce(aligned_size);
    }
    const int evict_num2 = Global<dtr::TensorPool>::Get()->num_forced_eviction();
    const auto duration = profiler::GetTimeNow() - started_at;
    search_free_mem_cost_.emplace_back(size, evict_num2 - evict_num1, duration);
    if (EnvBool<ONEFLOW_DTR_RECORD_MEM_FRAG_RATE>()) {
      size_t free_mem = 0;
      for (const auto& pair : ptr2piece_) {
        Piece* piece = pair.second;
        if (piece->is_free) {
          CHECK_ISNULL(piece->tensor);
          free_mem += piece->size;
        }
      }
      dtr::append_memory_frag_info_and_get(free_mem, dtr::memory_threshold());
    }
  }

  if (piece == nullptr) { DisplayAllPieces(); }

  CHECK(piece != nullptr) << "Error! : Out of memory when allocate size : " << size;
  CHECK_NOTNULL(piece->ptr);
  CHECK(ptr2piece_.find(piece->ptr) != ptr2piece_.end());
  if (dtr::is_enabled_and_debug()) {
    LOG(INFO) << "allocate offset: " << get_offset(piece->ptr) << ", size: " << piece->size
              << std::endl;
  }
  *mem_ptr = piece->ptr;
  total_allocate_bytes_ += size;

  if (EnvBool<ONEFLOW_DTR_OPERATION_LOG>()) {
    LOG(INFO) << "****"
              << "ALLOCATE-" << mem_ptr << "-" << size << std::endl;
  }

  // if (oneflow::DTRDebugEnabled()) {
  //   std::cout << "aid " << id_ << ", allocate " << (size / 1024. / 1024.)
  //             << "MB, total allocate bytes: " << (total_allocate_bytes_ / 1024. / 1024.)
  //             << std::endl;
  // }
}

void DtrCudaAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  if (mem_ptr == nullptr) { return; }

  auto it = ptr2piece_.find(mem_ptr);
  CHECK(it != ptr2piece_.end()) << "Error! : Try deallocate mem_ptr non-existent. mem ptr = "
                                << mem_ptr << " size = " << size;
  Piece* piece = it->second;
  CHECK_NOTNULL(piece);
  CHECK_EQ(piece->ptr, mem_ptr);
  CHECK(!piece->is_free);

  piece->is_free = true;
  piece->tensor = nullptr;
  piece->is_left = true;

  Piece* last_piece_insert_to_free_list = piece;
  Piece* next_p = piece->next;
  Piece* prev_p = piece->prev;

  if (next_p != nullptr && next_p->is_free) {
    CHECK_EQ(next_p->ptr, piece->ptr + piece->size);
    EraseFromFreeList(next_p);
    MergeNeighbourFreePiece(piece, next_p);
  }

  if (prev_p != nullptr && prev_p->is_free) {
    CHECK_EQ(piece->ptr, prev_p->ptr + prev_p->size);
    EraseFromFreeList(prev_p);
    MergeNeighbourFreePiece(prev_p, piece);
    last_piece_insert_to_free_list = prev_p;
  }
  InsertToFreeList(last_piece_insert_to_free_list);
  total_deallocate_bytes_ += size;
  // if (oneflow::DTRDebugEnabled()) {
  //   std::cout << "id: " << id_ << "deallocate " << (size / 1024. / 1024.)
  //             << "MB, total deallocate bytes: " << (total_deallocate_bytes_ / 1024. / 1024.)
  //             << std::endl;
  // }
}

size_t DtrCudaAllocator::allocated_memory() {
  CHECK_GE(total_allocate_bytes_, total_deallocate_bytes_);
  return total_allocate_bytes_ - total_deallocate_bytes_;
}

nlohmann::json DtrCudaAllocator::DumpSearchFreeMemCost() {
  return {{"overhead", search_free_mem_cost_}};
}

}  // namespace vm
}  // namespace oneflow

#endif
