#ifndef ONEFLOW_CORE_COMMON_RANGE_H_
#define ONEFLOW_CORE_COMMON_RANGE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class Range final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Range);
  Range() : Range(0, 0) {}
  ~Range() = default;

  Range(int64_t begin, int64_t end) : begin_(begin), end_(end) {}

  bool operator==(const Range& rhs) const {
    return begin_ == rhs.begin_ && end_ == rhs.end_;
  }

  int64_t begin() const { return begin_; }
  int64_t end() const { return end_; }

  int64_t& mut_begin() { return begin_; }
  int64_t& mut_end() { return end_; }

  int64_t size() const { return end_ - begin_; }

 private:
  int64_t begin_;
  int64_t end_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_RANGE_H_
