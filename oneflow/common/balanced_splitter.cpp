#include "oneflow/common/balanced_splitter.h"
#include "glog/logging.h"

namespace oneflow {

BalancedSplitter::BalancedSplitter(int64_t total_num, int64_t split_num) {
  size_per_range_ = total_num / split_num;
  change_pos_ = total_num % split_num;
  split_num_ = split_num;
}

Range BalancedSplitter::At(int64_t idx) const {
  CHECK_LT(idx, split_num_);
  int64_t lower_pound_num;
  int64_t upper_pound_num;
  if (idx < change_pos_) {
    lower_pound_num = (size_per_range_ + 1) * idx;
    upper_pound_num = lower_pound_num + (size_per_range_ + 1);
  } else {
    lower_pound_num = (size_per_range_ + 1) * change_pos_
      + size_per_range_ * (idx - change_pos_);
    upper_pound_num = lower_pound_num + size_per_range_;
  }
  return Range(lower_pound_num, upper_pound_num);
}

} // namespace oneflow
