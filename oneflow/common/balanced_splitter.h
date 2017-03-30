#ifndef ONEFLOW_COMMON_BALANCED_SPLITTER_H_
#define ONEFLOW_COMMON_BALANCED_SPLITTER_H_

#include <stdint.h>
#include "common/util.h"

namespace oneflow {

// For example
// BalancedSplitter splitter(20, 6)
// the result of splitter.At 0,1,2,3,4,5 is
//                           4,4,3,3,3,3
class BalancedSplitter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BalancedSplitter);
  
  BalancedSplitter() = default;
  ~BalancedSplitter() = default;

  void Init(int64_t total_num, int64_t split_num);

  int64_t At(int64_t idx) const;

 private:
  int64_t minimum_guarantee_;
  int64_t threshold_;

};

} // namespace oneflow

#endif // ONEFLOW_COMMON_BALANCED_SPLITTER_H_
