#include "oneflow/core/common/range.h"

namespace oneflow {

Range::Range(const RangeProto& range_proto){
    begin_ = range_proto.begin();
    end_ = range_proto.end();
}

void Range::ToProto(RangeProto* ret) const {
  ret->set_begin(begin_);
  ret->set_end(end_);
}

Range FindIntersectant(const Range& lhs, const Range& rhs) {
  if (lhs.end() > rhs.begin() && rhs.end() > lhs.begin()) {
    int64_t left = lhs.begin() > rhs.begin() ? lhs.begin() : rhs.begin();
    int64_t right = lhs.end() < rhs.end() ? lhs.end() : rhs.end();
    return Range(left, right);
  } else {
    return Range(0, 0);
  }
}

}  // namespace oneflow