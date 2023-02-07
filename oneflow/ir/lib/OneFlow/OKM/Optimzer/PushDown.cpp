#include "OneFlow/OKM/Optimizer/PushDown.h"
#include "oneflow/core/common/util.h"

namespace {
void AlignSize(int64_t& size) {
  if (size % (oneflow::kBlobBodyAlignSize)) {
    size = (size / (oneflow::kBlobBodyAlignSize) + 1) * oneflow::kBlobBodyAlignSize;
  }
}

}  // namespace

namespace mlir {
namespace okm {

void MemoryList::PushDown(int64_t size) {
  AlignSize(size);
  std::list<MemoryItem>::iterator it = list_.begin(), best_fit = list_.end(), max_it = list_.end();
  while (it != list_.end()) {
    if (!it->used_) {
      if (max_it == list_.end() || it->size_ > max_it->size_) { max_it = it; }
      if ((best_fit == list_.end() && it->size_ >= size)
          || (it->size_ >= size && it->size_ < best_fit->size_)) {
        best_fit = it;
      }
    }
    ++it;
  };
  uint64_t ptr = 0;
  if (best_fit != list_.end()) {
    list_.insert(best_fit, MemoryItem{best_fit->offset_, size, true});
    if (auto remain_size = best_fit->size_ - size) {
      list_.insert(best_fit, MemoryItem{best_fit->offset_ + size, remain_size, false});
    }
    ptr = best_fit->offset_;
    list_.erase(best_fit);
  } else if (max_it != list_.end()) {
    auto expand = size - max_it->size_;
    it = list_.insert(max_it, MemoryItem{max_it->offset_, size, true});
    it++;
    it++;
    while (it != list_.end()) {
      it->offset_ += expand;
      ++it;
    }
    for (auto& elem : ptr_vec_) {
      if (elem > max_it->offset_) { elem += expand; }
    }
    ptr = max_it->offset_;
    total_size_ += expand;
    it = list_.erase(max_it);
  } else {
    list_.push_back(MemoryItem{total_size_, size, true});
    ptr = total_size_;
    total_size_ += size;
  }
  ptr_vec_.push_back(ptr);
}

void MemoryList::FreeMem(int idx) {
  if (idx > ptr_vec_.size()) { LOG(FATAL) << "Index is out of range in PushDown Algo"; }
  auto ptr = ptr_vec_[idx];
  std::list<MemoryItem>::iterator it = list_.begin();
  while (it != list_.end()) {
    if (it->offset_ == ptr) {
      if (!it->used_) { LOG(FATAL) << "Double Free in PushDown Algo"; }
      break;
    }
    it++;
  }
  if (it == list_.end()) { LOG(FATAL) << "Failed to find mem item"; }
  if (it != list_.begin()) {
    std::list<MemoryItem>::iterator prev_it = it;
    prev_it--;
    if (prev_it->used_) {
      it->used_ = false;
    } else {
      list_.insert(prev_it, MemoryItem{prev_it->offset_, prev_it->size_ + it->size_, false});
      list_.erase(prev_it);
      list_.erase(it);
    }
  } else {
    it->used_ = false;
  }
}
}  // namespace okm
}  // namespace mlir