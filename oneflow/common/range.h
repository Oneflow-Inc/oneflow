#ifndef ONEFLOW_COMMON_RANGE_H_
#define ONEFLOW_COMMON_RANGE_H_

#include "common/util.h"

namespace oneflow {

class Range final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Range);
  Range() : Range(0, 0) {}
  ~Range() = default;

  Range(int32_t begin, int32_t end) : begin_(begin), end_(end) {}

  int32_t begin() const { return begin_; }
  int32_t end() const { return end_; }
  
  int32_t& mut_begin() { return begin_; }
  int32_t& mut_end() { return end_; }

  int32_t size() const { return end_ - begin_; }

 private:
  int32_t begin_;
  int32_t end_;
};

} // namespace oneflow

#endif // ONEFLOW_COMMON_RANGE_H_
