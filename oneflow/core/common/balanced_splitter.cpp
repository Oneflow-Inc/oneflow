#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

BalancedSplitter::BalancedSplitter(int64_t total_num, int64_t split_num) {
  base_part_size_ = total_num / split_num;
  base_begin_idx_ = total_num % split_num;
  split_num_ = split_num;
}

Range BalancedSplitter::At(int64_t idx) const {
  CHECK_LT(idx, split_num_);
  int64_t left_bound = -1;
  int64_t right_bound = -1;
  if (idx < base_begin_idx_) {
    left_bound = (base_part_size_ + 1) * idx;
    right_bound = left_bound + (base_part_size_ + 1);
  } else {
    left_bound = (base_part_size_ + 1) * base_begin_idx_
                 + base_part_size_ * (idx - base_begin_idx_);
    right_bound = left_bound + base_part_size_;
  }
  return Range(left_bound, right_bound);
}

Range BalancedSplitter::At(int64_t first_idx, int64_t last_idx) const {
  CHECK_LE(first_idx, last_idx);
  CHECK_LT(last_idx, split_num_);
  Range first_range = At(first_idx);
  Range last_range = At(last_idx);
  return Range(first_range.begin(), last_range.end());
}

}  // namespace oneflow
