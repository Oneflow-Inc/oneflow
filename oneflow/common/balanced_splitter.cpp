#include "balanced_splitter.h"
#include "glog/logging.h"

namespace oneflow {

BalancedSplitter::BalancedSplitter(int64_t total_num, int64_t split_num) {
  CHECK_GE(total_num, split_num);
  range_per_size_ = total_num / split_num;
  change_point_ = total_num % split_num;
  split_num_ = split_num;
}

Range BalancedSplitter::At(int64_t idx) const {
  CHECK_LT(idx, split_num_);
  int64_t lower_pound_num;
  int64_t upper_pound_num;
  if (idx < change_point_) {
    lower_pound_num = (range_per_size_ + 1) * idx;
    upper_pound_num = lower_pound_num + (range_per_size_ + 1);
  } else {
    lower_pound_num = (range_per_size_ + 1) * change_point_
      + range_per_size_ * (idx - change_point_);
    upper_pound_num = lower_pound_num + range_per_size_;
  }
  return Range(lower_pound_num, upper_pound_num);
}

} // namespace oneflow
