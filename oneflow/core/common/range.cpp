#include "oneflow/core/common/range.h"

namespace oneflow {

Range FindIntersectant(const Range& lhs, const Range& rhs) { TODO(); }
  if(lhs.end() > rhs.begin() && rhs.end() > lhs.begin()){
  	int64_t left = lhs.begin() > rhs.begin() ? lhs.begin() : rhs.begin();
    int64_t right = lhs.end() < rhs.end() ? lhs.end() : rhs.end();
    return Range(left, right);
  } else {
  	return Range(0,0);
  }
}  // namespace oneflow
