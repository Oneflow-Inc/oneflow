#ifndef ONEFLOW_BALANCED_SPLITTER_H_
#define ONEFLOW_BALANCED_SPLITTER_H_

#include <stdint.h>

namespace oneflow {

// BalancedSplitter splitter(20, 6)
// the result of splitter.at 0,1,2,3,4,5 is
//                           4,4,3,3,3,3
class BalancedSplitter {
 public:
  BalancedSplitter() = default;
  BalancedSplitter(const BalancedSplitter&) = delete;
  BalancedSplitter(BalancedSplitter&&) = delete;
  BalancedSplitter& operator(const BalancedSplitter&) = delete;
  BalancedSplitter& operator(BalancedSplitter&&) = delete;
  ~BalancedSplitter() = default;

  void init(int64_t total_num, int64_t split_num);

  void at(int64_t idx) const;

 private:
  int64_t minimum_guarantee_;
  int64_t threshold_;

};

} // namespace oneflow

#endif // ONEFLOW_BALANCED_SPLITTER_H_
