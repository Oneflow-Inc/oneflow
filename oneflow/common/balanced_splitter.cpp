#include "balanced_splitter.h"
#include "glog/logging.h"

namespace oneflow {

BalancedSplitter::BalancedSplitter(int64_t total_num, int64_t split_num) {
  CHECK_GE(total_num, split_num);
  int64_t quotient_num = total_num / split_num;
  int64_t remainder_num = total_num % split_num;
  int64_t lower_bound_num = 0;
  for (int64_t i = 0; i < split_num; ++i) {
    int64_t upper_bound_num = lower_bound_num + quotient_num;
    if (i < remainder_num) {
      upper_bound_num++;
    }
    splited_ranges_.push_back(Range(lower_bound_num, upper_bound_num));
    lower_bound_num = upper_bound_num;
  }
  CHECK_EQ(lower_bound_num, total_num);
}

Range BalancedSplitter::At(int64_t idx) const {
  CHECK_LT(idx, splited_ranges_.size());
  return splited_ranges_[idx];
}

} // namespace oneflow
