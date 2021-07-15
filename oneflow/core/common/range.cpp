/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/range.h"

namespace oneflow {

Range::Range(const RangeProto& range_proto) {
  begin_ = range_proto.begin();
  end_ = range_proto.end();
}

void Range::ToProto(RangeProto* ret) const {
  ret->set_begin(begin_);
  ret->set_end(end_);
}

Maybe<void> Range::ForEachSubRange(
    int64_t sub_range_size, const std::function<Maybe<void>(const Range&)>& DoEachRange) const {
  CHECK_EQ_OR_RETURN(size() % sub_range_size, 0);
  int64_t start = begin();
  for (; start < end(); start += sub_range_size) {
    JUST(DoEachRange(Range(start, start + sub_range_size)));
  }
  CHECK_EQ_OR_RETURN(start, end());
  return Maybe<void>::Ok();
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
