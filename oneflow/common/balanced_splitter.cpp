#include "balanced_splitter.h"
#include "glog/logging.h"

namespace oneflow {

void BalancedSplitter::Init(int64_t total_num, int64_t split_num) {
  CHECK_GE(total_num, split_num);
  CHECK_GT(split_num, 0);
  minimum_guarantee_ = total_num / split_num;
  threshold_ = total_num % split_num;
}

int64_t BalancedSplitter::At(int64_t idx) const {
  CHECK_GE(idx, 0);
  if (idx < threshold_) {
    return minimum_guarantee_ + 1;
  } else {
    return minimum_guarantee_;
  }
}

} // namespace oneflow
