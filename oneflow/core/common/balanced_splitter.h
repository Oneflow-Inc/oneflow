#ifndef ONEFLOW_CORE_COMMON_BALANCED_SPLITTER_H_
#define ONEFLOW_CORE_COMMON_BALANCED_SPLITTER_H_

#include <stdint.h>
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

// For example
// BalancedSplitter splitter(20, 6)
// the result of splitter.At
//     0    [0, 4)
//     1    [4, 8)
//     2    [8, 11)
//     3    [11, 14)
//     4    [14, 17)
//     5    [17, 20)
class BalancedSplitter final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BalancedSplitter);
  BalancedSplitter() = delete;
  ~BalancedSplitter() = default;

  BalancedSplitter(int64_t total_num, int64_t split_num);

  Range At(int64_t idx) const;

  int64_t BaseBeginIdx() const { return change_pos_; }
  int64_t BasePartSize() const { return size_per_range_; }
  int64_t BiggerPartSize() const { return size_per_range_ + 1; }

 private:
  int64_t size_per_range_;
  int64_t change_pos_;
  int64_t split_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BALANCED_SPLITTER_H_
