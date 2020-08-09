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