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
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/common/thread_local_guard.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/profiler/util.h"

#include "oneflow/core/common/env_var/remat.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/vm/remat/allocator.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/remat/env.h"
#include "oneflow/core/vm/remat/util.h"
#include "oneflow/core/vm/thread_safe_guard.h"
#include "oneflow/core/vm/remat/disjoint_set.h"
#include <iostream>

namespace oneflow {
namespace vm {

namespace {

inline size_t CudaMemAlignedBytes(size_t bytes) { return RoundUp(bytes, kCudaMemAllocAlignSize); }

inline bool IsAlignedSize(size_t size) { return size % kCudaMemAllocAlignSize == 0; }

inline double bytes2Mb(size_t bytes) { return bytes * 1. / 1024 / 1024; }

static constexpr size_t kSmallPieceThreshold = 10 * 1024;  // 10 KB

inline bool ShouldBeHeldBySmallPiece(size_t size) {
  return Singleton<remat::Env>::Get()->is_small_pieces_optimization_enabled()
         && size <= kSmallPieceThreshold;
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

RematEpAllocator::RematEpAllocator(size_t alignment, std::unique_ptr<Allocator>&& backend)
    : Allocator(),
      alignment_(alignment),
      backend_(std::move(backend)),
      memory_size_(0),
      recycle_piece_list_(nullptr),
      normal_group_num_(EnvInteger<ONEFLOW_REMAT_GROUP_NUM>()),
      group_indexes_(GroupNumToIndexes(normal_group_num_)),
      cur_group_index_id_(normal_group_num_ > 1 ? 1 : 0),
      cur_group_index_id_high_cost_(0),
      enable_left_and_right_(normal_group_num_ > 1) {
  free_pieces_overlapping_with_group_.resize(normal_group_num_ + 1);
}

RematEpAllocator::~RematEpAllocator() {
  if (memory_ != nullptr) { backend_->Deallocate(static_cast<char*>(memory_), memory_size_); }
}

RematEpAllocator::offset_t RematEpAllocator::get_offset(const char* mem_ptr) const {
  return mem_ptr - (char*)memory_;
}

void RematEpAllocator::LinkStorageAndPtr(RematableTensorStorage* storage, const char* mem_ptr) {
  Piece* piece = ptr2piece_.at(mem_ptr);
  piece->tensor = storage;
  CHECK_NOTNULL(piece->tensor);
  VLOG(1) << "tensor " << piece->tensor->id() << " is allocated at " << get_offset(mem_ptr)
          << ", left: " << piece->is_left;
}

Maybe<bool> RematEpAllocator::InSmallMemoryArea(void* ptr) {
  CHECK_NOTNULL_OR_RETURN(small_piece_area_ptr_);
  CHECK_GE_OR_RETURN(ptr, memory_);
  CHECK_LT_OR_RETURN(ptr, (char*)memory_ + memory_size_);
  // compare pointer by raw < or > is undefined behavior
  return std::greater_equal<>{}(ptr, small_piece_area_ptr_);
}

RematEpAllocator::Piece* RematEpAllocator::AllocatePiece() {
  if (recycle_piece_list_) {
    Piece* ret = recycle_piece_list_;
    recycle_piece_list_ = recycle_piece_list_->next;
    return ret;
  } else {
    pieces_.emplace_back(new Piece());
    return pieces_.at(pieces_.size() - 1).get();
  }
}

void RematEpAllocator::DeallocatePiece(Piece* piece) {
  piece->ptr = nullptr;
  piece->size = 0;
  CHECK(piece->is_free);
  piece->prev = nullptr;
  piece->next = recycle_piece_list_;
  piece->is_left = true;
  recycle_piece_list_ = piece;
}

void RematEpAllocator::InsertPiece2PtrMap(Piece* piece) {
  VLOG(2) << "insert piece, offset " << get_offset(piece->ptr);
  CHECK_NOTNULL(piece->ptr);
  CHECK(ptr2piece_.emplace(piece->ptr, piece).second);
}

void RematEpAllocator::ErasePieceFromPtrMap(Piece* piece) {
  VLOG(2) << "erase piece, offset " << get_offset(piece->ptr);
  CHECK_NOTNULL(piece->ptr);
  auto it = ptr2piece_.find(piece->ptr);
  CHECK(it != ptr2piece_.end());
  ptr2piece_.erase(it);
}

double get_cost(const vm::RematableTensorStorage* storage) {
  if (storage == nullptr) { return 0.; }
  double cost = CHECK_JUST(storage->cost(0));

  CHECK(!std::isnan(cost));
  return cost;
}

double get_cost(const vm::RematableTensorStorage* storage, size_t size) {
  if (storage == nullptr) { return 0.; }
  double cost = CHECK_JUST(storage->cost(size));

  CHECK(!std::isnan(cost));
  return cost;
}

void RematEpAllocator::CheckPieces() {
  auto it = ptr2piece_.cbegin();
  for (int i = 0; i < ptr2piece_.size(); ++i) {
    Piece* piece = it->second;
    if (piece->tensor == nullptr) { CHECK(piece->is_free); }
    if (piece->is_free) { CHECK_ISNULL(piece->tensor); }
    if (i != 0) {
      CHECK_EQ(piece->prev->next, piece);
      CHECK_EQ(piece->prev->ptr + piece->prev->size, piece->ptr);
      auto it2 = it;
      --it2;
      CHECK_EQ(piece->prev, it2->second);
    }
    if (i != ptr2piece_.size() - 1) {
      CHECK_EQ(piece->next->prev, piece);
      CHECK_EQ(piece->ptr + piece->size, piece->next->ptr);
      auto it2 = it;
      ++it2;
      CHECK_EQ(piece->next, it2->second);
    }
    it++;
  }
}

void RematEpAllocator::DisplayAllPieces() {
  std::cout << "ops: " << Singleton<remat::Env>::Get()->ops.size() << std::endl;
  for (const auto& pair : ptr2piece_) {
    Piece* piece = pair.second;
    std::stringstream ss;
    ss << "piece " << piece << ", " << (void*)piece->ptr << ", " << piece->size << ", ";
    if (piece->tensor) {
      ss << "ebo: " << piece->tensor << ", id: " << piece->tensor->id() << ", cost: "
         << (piece->tensor->is_eviction_disabled() ? "disabled"
                                                   : std::to_string(get_cost(piece->tensor)))
         << ", pinned: " << piece->tensor->num_pinned()
         << ", evictable: " << piece->tensor->is_evictable()
         << ", compute op: " << piece->tensor->compute_op_type_name();
    } else {
      ss << "no tensor";
    }
    std::cout << ss.str() << std::endl;
  }
}

void RematEpAllocator::Display() {
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

RematEpAllocator::offset_t RematEpAllocator::FindProperPositionInGroup(Piece* piece,
                                                                       size_t group_idx,
                                                                       size_t request_size) const {
  const offset_t grp_left_bound = group_boundaries_[group_idx].first;
  const offset_t grp_right_bound = group_boundaries_[group_idx].second;
  const offset_t piece_left_bound = get_offset(piece->ptr);
  const offset_t piece_right_bound = piece_left_bound + piece->size;
  const bool is_right =
      enable_left_and_right_ && (group_idx % 2 == 1) && group_idx != normal_group_num_;
#define PNT3(var) VLOG(3) << OF_PP_STRINGIZE(var) << ": " << var << std::endl
  PNT3(group_idx);
  PNT3(grp_left_bound);
  PNT3(grp_right_bound);
  PNT3(piece_left_bound);
  PNT3(piece_right_bound);
  PNT3(is_right);
  PNT3(request_size);

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

void RematEpAllocator::InsertToFreeList(Piece* piece) {
  const offset_t piece_left = get_offset(piece->ptr);
  const offset_t piece_right = piece_left + piece->size;
  VLOG(3) << "piece_left: " << piece_left << ", right: " << piece_right << std::endl;
  for (size_t i = 0; i < group_boundaries_.size(); i++) {
    VLOG(3) << "g left: " << group_boundaries_[i].first
            << ", right: " << group_boundaries_[i].second << std::endl;
    if ((piece_left >= group_boundaries_[i].first && piece_left < group_boundaries_[i].second)
        || (piece_right > group_boundaries_[i].first
            && piece_right <= group_boundaries_[i].second)) {
      VLOG(3) << "overlap" << std::endl;
      free_pieces_overlapping_with_group_[i].insert(piece);
    }
  }
}

void RematEpAllocator::EraseFromFreeList(Piece* piece) {
  VLOG(3) << "erase " << get_offset(piece->ptr);
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

auto RematEpAllocator::AllocateMemoryInPiece(Piece* piece, offset_t offset_in_piece, size_t size)
    -> Piece* {
  auto SplitPiece = [this](Piece* piece, offset_t offset_in_piece) -> Piece* {
    // offset_in_piece must be less (not equal) than piece->size so that
    // new_piece has size
    CHECK_LE(offset_in_piece, piece->size);
    Piece* new_piece = AllocatePiece();
    new_piece->ptr = piece->ptr + offset_in_piece;
    VLOG(2) << get_offset(piece->ptr);
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
  // piece2->is_free = false;
  if (piece3 != nullptr) {
    piece3->is_free = true;
    InsertToFreeList(piece3);
  }
  return piece2;
}

size_t RematEpAllocator::iterate_group_index(bool high) const {
  if (normal_group_num_ == 1) { return 0; }
  auto is_high_group = [](size_t idx) -> bool { return (idx / 2) % 2 == (idx % 2); };
  if (high) {
    size_t index;  // NOLINT
    do {
      cur_group_index_id_high_cost_ = (cur_group_index_id_high_cost_ + 1) % normal_group_num_;
      index = group_indexes_[cur_group_index_id_high_cost_];
    } while (!is_high_group(index));
    return index;
  } else {
    size_t index;  // NOLINT
    do {
      cur_group_index_id_ = (cur_group_index_id_ + 1) % normal_group_num_;
      index = group_indexes_[cur_group_index_id_];
    } while (is_high_group(index));
    return index;
  }
}

size_t RematEpAllocator::group_index(bool high) const {
  if (high) {
    return group_indexes_[cur_group_index_id_high_cost_];
  } else {
    return group_indexes_[cur_group_index_id_];
  }
}

void RematEpAllocator::InitMemory() {
  memory_size_ = Singleton<remat::Env>::Get()->budget_in_bytes();
  CHECK_JUST(backend_->Allocate(&memory_, memory_size_));
  LOG(INFO) << "memory_: " << (void*)memory_ << ", size: " << memory_size_;
  const size_t small_piece_area_size =
      Singleton<remat::Env>::Get()->is_small_pieces_optimization_enabled()
          ? 1024 * kSmallPieceThreshold
          : 0;
  const size_t normal_area_size = memory_size_ - small_piece_area_size;
  small_piece_area_ptr_ = memory_ + normal_area_size;

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
  piece->ptr = memory_;
  piece->prev = nullptr;
  piece->next = nullptr;
  piece->is_free = true;
  piece->tensor = nullptr;
  InsertToFreeList(piece);
  InsertPiece2PtrMap(piece);
}

Maybe<RematEpAllocator::Piece*> RematEpAllocator::FindPiece(size_t aligned_size,
                                                            bool after_eviction) {
  CHECK_OR_RETURN(IsAlignedSize(aligned_size));

  if (memory_ == nullptr) { InitMemory(); }

  // NOLINTNEXTLINE
  const bool is_high_op = [&]() {
    std::vector<std::string> high_compute_cost_names{"conv2d", "conv_data_grad", "conv_filter_grad",
                                                     "add_n",  "matmul",         "batch_matmul"};
    const auto current_op_type_name =
        CHECK_JUST(ThreadLocalGuard<remat::CurrentOpTypeName>::Current())->value;
    PNT3(current_op_type_name);
    if (std::find(high_compute_cost_names.cbegin(), high_compute_cost_names.cend(),
                  current_op_type_name)
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
  PNT3(aligned_size);
  size_t iterate_num = 0;
  do {
    const auto& free_pieces = free_pieces_overlapping_with_group_[group_idx];
    PNT3(group_idx);
    PNT3(free_pieces.size());
    for (auto it = free_pieces.begin(); it != free_pieces.end(); ++it) {
      Piece* piece = *it;
      CHECK_OR_RETURN(piece->is_free);
      CHECK_NOTNULL(piece->ptr);
      CHECK_OR_RETURN(IsAlignedSize(piece->size));
      PNT3(get_offset(piece->ptr));
      PNT3(piece->size);
      if (piece->size >= aligned_size) {
        const offset_t offset_in_memory = FindProperPositionInGroup(piece, group_idx, aligned_size);
        PNT3(offset_in_memory);
        if (offset_in_memory != SIZE_MAX) {
          const offset_t offset_in_piece = offset_in_memory - get_offset(piece->ptr);
          auto ret = AllocateMemoryInPiece(piece, offset_in_piece, aligned_size);
          CheckPieces();
          return ret;
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

void RematEpAllocator::MergeNeighbourFreePiece(Piece* lhs, Piece* rhs) {
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

Maybe<RematEpAllocator::Piece*> RematEpAllocator::EvictAndFindPieceLoop(size_t required_size,
                                                                        bool consider_neighbor) {
  VLOG(2) << "required size: " << required_size;
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

  while (true) {
    double min_cost = std::numeric_limits<double>::max();
    vm::RematableTensorStorage* min_tensor = nullptr;
    for (auto it = ptr2piece_.begin();
         it != ptr2piece_.end() && !JUST(InSmallMemoryArea(it->second->ptr)); it++) {
      auto* tensor = it->second->tensor;
      if (tensor != nullptr && !tensor->is_pinned() && tensor->is_evictable()) {
        auto cur_op_cost =
            consider_neighbor ? get_cost(
                tensor, GetSizeIncludingNeighborhood(it, ptr2piece_.begin(), ptr2piece_.end()))
                              : get_cost(tensor);
        if (cur_op_cost < min_cost) {
          min_cost = cur_op_cost;
          min_tensor = tensor;
        }
      }
    }
    if (min_tensor) {
      min_tensor->Evict(false);
      Piece* piece = JUST(FindPiece(required_size, true));
      if (piece != nullptr) { return piece; }
    } else {
      return Error::RuntimeError() << "Cannot find a piece to evict";
    }
  }
}

Maybe<RematEpAllocator::Piece*> RematEpAllocator::EvictAndFindPieceOnce(size_t required_size) {
  VLOG(2) << "required size: " << required_size;
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
  while (end != ptr2piece_.end() && !JUST(InSmallMemoryArea(end->second->ptr))) {
    if (total_size < required_size) {
      auto* end_tensor = end->second->tensor;
      if (end_tensor != nullptr && (end_tensor->is_pinned() || !end_tensor->is_evictable())) {
        VLOG(2) << "skip tensor: " << end_tensor << ", size: " << end_tensor->blob_bytes()
                << ", compute op " << end_tensor->compute_op_type_name()
                << ", num_pinned: " << end_tensor->num_pinned()
                << ", is_evictable: " << end_tensor->is_evictable();
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
      auto cur_op_cost = get_cost(end_tensor);
      costs.push_back(cur_op_cost);
      cost_except_size += cur_op_cost;
      VLOG(2) << "move end, include op: "
              << (end_tensor != nullptr ? end_tensor->compute_op_type_name() : "no tensor")
              << ", size: " << end->second->size << ", total_size: " << total_size
              << ", total cost: " << cost_except_size << ", cur op cost: " << cur_op_cost;
      end++;
      end_i++;
    } else {
      auto* start_tensor = start->second->tensor;
      // const auto* start_tensor = start->second->tensor;
      total_size -= start->second->size;
      // start_tensor is back in the pool, update_after_pesudo_compute
      double cur_op_cost = 0;
      cur_op_cost = costs[start_i];
      cost_except_size -= cur_op_cost;
      VLOG(2) << "move start, exclude op: "
              << (start_tensor != nullptr ? start_tensor->compute_op_type_name() : "no tensor")
              << ", size: " << start->second->size << ", total_size: " << total_size
              << ", total cost: " << cost_except_size << ", cur op cost: " << cur_op_cost;
      start++;
      start_i++;
    }
    double cost = cost_except_size;
    if (total_size >= required_size && cost < min_cost) {
      min_cost = cost;
      min_start = start;
      min_end = end;
      VLOG(2) << "record, min_cost: " << min_cost;
    }
  }
  // CHECK(min_end != start);
  // collect piece ptrs into a new container, because evict() will devalidate the iterators
  std::vector<Piece*> pieces_to_be_evicted;
  for (auto it = min_start; it != min_end; ++it) {
    Piece* piece = it->second;
    pieces_to_be_evicted.push_back(piece);
  }
  if (IsInDebugMode()) {
    for (auto* piece : pieces_to_be_evicted) {
      LOG(INFO) << "release dptr: " << get_offset(piece->ptr) << ", size: " << piece->size
                << ", cost: " << get_cost(piece->tensor) << ", compute op: "
                << (piece->tensor != nullptr ? piece->tensor->compute_op_type_name() : "no")
                << ", id: "
                << (piece->tensor != nullptr ? std::to_string(piece->tensor->id()) : "no");
    }
  }
  size_t evict_size = 0;
  for (auto* piece : pieces_to_be_evicted) {
    evict_size += piece->size;
    // NOTE: evict will trigger the merge and deallocation of neighbour free pieces,
    // e.g. two contiguous pieces relu, no_tensor, after relu evict, no_tensor will be deallocated.
    // currently deallocation only set tensor to nullptr, not real free,
    // so no bug occurs. It is tricky and fragile.
    if (piece->tensor != nullptr) {
      CHECK_OR_RETURN(!ShouldBeHeldBySmallPiece(piece->size));
      piece->tensor->Evict(false);
    }
  }
  VLOG(2) << "evict size: " << evict_size;

  if (!pieces_to_be_evicted.empty()) { return CHECK_NOTNULL(JUST(FindPiece(required_size, true))); }
  return nullptr;
}

Maybe<void> RematEpAllocator::Allocate(char** mem_ptr, std::size_t size) {
  if (size == 0) {
    *mem_ptr = nullptr;
    return Maybe<void>::Ok();
  }
  ReentrantThreadSafeLock::RAIIGuard guard(thread_lock_);
  size_t aligned_size = CudaMemAlignedBytes(size);

  Piece* piece = JUST(FindPiece(aligned_size, false));

  if (piece == nullptr) {
    if (first_time) {
      if (EnvBool<ONEFLOW_REMAT_DISPLAY_IN_FIRST_TIME>()) { DisplayAllPieces(); }
      first_time = false;
    }
    const auto started_at = profiler::GetTimeNow();
    const size_t evict_num1 = Singleton<remat::Env>::Get()->forced_eviction_num();
    if (EnvBool<ONEFLOW_REMAT_HEURISTIC_DTE>()) {
      piece = JUST(EvictAndFindPieceLoop(aligned_size, true));
    } else if (EnvBool<ONEFLOW_REMAT_HEURISTIC_DTR>()) {
      piece = JUST(EvictAndFindPieceLoop(aligned_size, false));
    } else {
      piece = JUST(EvictAndFindPieceOnce(aligned_size));
    }
    const size_t evict_num2 = Singleton<remat::Env>::Get()->forced_eviction_num();
    const auto duration = profiler::GetTimeNow() - started_at;
    search_free_mem_cost_.emplace_back(size, evict_num2 - evict_num1, duration);
    if (EnvBool<ONEFLOW_REMAT_RECORD_MEM_FRAG_RATE>()) {
      size_t free_mem = 0;
      for (const auto& pair : ptr2piece_) {
        Piece* piece = pair.second;
        if (piece->is_free) {
          CHECK_ISNULL_OR_RETURN(piece->tensor);
          free_mem += piece->size;
        }
      }
      remat::append_memory_frag_info_and_get(free_mem, memory_size_);
    }
  }

  if (piece == nullptr) { DisplayAllPieces(); }

  CHECK_OR_RETURN(piece != nullptr) << "Error! : Out of memory when allocate size : " << size;
  CHECK_NOTNULL(piece->ptr);
  CHECK_OR_RETURN(ptr2piece_.find(piece->ptr) != ptr2piece_.end());
  LOG(INFO) << "allocate offset: " << get_offset(piece->ptr) << ", size: " << piece->size
            << std::endl;
  *mem_ptr = piece->ptr;
  total_allocate_bytes_ += size;
  piece->is_free = false;

  return Maybe<void>::Ok();
}

void RematEpAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  if (mem_ptr == nullptr) { return; }
  ReentrantThreadSafeLock::RAIIGuard guard(thread_lock_);

  auto it = ptr2piece_.find(mem_ptr);
  CHECK(it != ptr2piece_.end()) << "Error! : Try deallocate mem_ptr non-existent. mem ptr = "
                                << mem_ptr << " size = " << size;
  Piece* piece = it->second;
  CHECK_NOTNULL(piece);
  CHECK_EQ(piece->ptr, mem_ptr);
  CHECK(!piece->is_free);

  if (auto* tensor = piece->tensor) {
    CHECK_JUST(remat::DisjointSet::update_after_release(tensor));
  }

  piece->is_free = true;
  piece->tensor = nullptr;
  piece->is_left = true;

  Piece* last_piece_insert_to_free_list = piece;
  Piece* next_p = piece->next;
  Piece* prev_p = piece->prev;

  VLOG(2) << "deallocate offset: " << get_offset(piece->ptr) << ", size: " << piece->size
          << ", prev: " << prev_p << ", next: " << next_p;

  if (next_p != nullptr && next_p->is_free) {
    CHECK_EQ(next_p->ptr, piece->ptr + piece->size);
    EraseFromFreeList(next_p);
    VLOG(2) << "merge with next_p";
    MergeNeighbourFreePiece(piece, next_p);
  }

  if (prev_p != nullptr && prev_p->is_free) {
    CHECK_EQ(piece->ptr, prev_p->ptr + prev_p->size);
    EraseFromFreeList(prev_p);
    VLOG(2) << "merge with prev_p";
    MergeNeighbourFreePiece(prev_p, piece);
    last_piece_insert_to_free_list = prev_p;
  }
  InsertToFreeList(last_piece_insert_to_free_list);
  total_deallocate_bytes_ += size;
  CheckPieces();
}

size_t RematEpAllocator::allocated_memory() {
  CHECK_GE(total_allocate_bytes_, total_deallocate_bytes_);
  return total_allocate_bytes_ - total_deallocate_bytes_;
}

void RematEpAllocator::DeviceReset() {
  ReentrantThreadSafeLock::RAIIGuard guard(thread_lock_);
  backend_->DeviceReset();
}

nlohmann::json RematEpAllocator::DumpSearchFreeMemCost() {
  return {{"overhead", search_free_mem_cost_}};
}

}  // namespace vm

vm::RematEpAllocator* remat::AllocatorManager::CreateOrGetAllocator(DeviceType device_type,
                                                                    size_t device_index) {
  auto key = std::make_pair(device_type, device_index);
  auto it = allocators_.find(key);
  if (it == allocators_.end()) {
    auto ep_device =
        Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
    auto ep_backend_allocator =
        std::make_unique<vm::EpBackendAllocator>(ep_device, ep::AllocationOptions{});
    auto allocator = std::make_unique<vm::RematEpAllocator>(ep::kMaxAlignmentRequirement,
                                                            std::move(ep_backend_allocator));
    allocators_.emplace(key, std::move(allocator));
    return allocators_.at(key).get();
  } else {
    return it->second.get();
  }
}

}  // namespace oneflow
