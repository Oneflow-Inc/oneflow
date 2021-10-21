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
#include "oneflow/core/ndarray/slice.h"

namespace oneflow {

Slice::Slice(const std::initializer_list<int64_t>& l) {
  DimVector vec(l);
  value_capacity_ = 0;
  if (vec.size() == 0) {
    start_ = kStart;
    end_ = kEnd;
    stride_ = 1;
  } else if (vec.size() == 1) {
    start_ = vec[0];
    end_ = kEnd;
    stride_ = 1;
  } else if (vec.size() == 2) {
    start_ = vec[0];
    end_ = vec[1];
    stride_ = 1;
  } else if (vec.size() == 3) {
    start_ = vec[0];
    end_ = vec[1];
    stride_ = vec[2];
  } else {
    UNIMPLEMENTED();
  }
}

bool Slice::IsBounded() const {
  CHECK_NE(stride_, 0);
  if (value_capacity_ == 0) { return false; }
  return (start_ >= 0) && (start_ <= value_capacity_ - (stride_ < 0)) && (end_ >= 0 - (stride_ < 0))
         && (end_ <= value_capacity_);
}

const Slice& Slice::Bound(size_t value_capacity) {
  CHECK_GT(value_capacity, 0);
  if (value_capacity_ == value_capacity) { return *this; }
  CHECK_EQ(value_capacity_, 0);
  value_capacity_ = value_capacity;
  if (start_ != kStart && start_ < 0) { start_ += value_capacity_; }
  if (end_ != kStart && end_ < 0) { end_ += value_capacity_; }
  if (start_ == kStart) { start_ = 0; }
  if (end_ == kEnd) { end_ = value_capacity_; }
  if (start_ == kEnd) { start_ = value_capacity_ - (stride_ < 0); }
  if (end_ == kStart) { end_ = 0 - (stride_ < 0); }
  CHECK_NE(stride_, 0);
  CHECK_GE(start_, 0);
  CHECK_LE(start_, value_capacity_);
  CHECK_GE(end_, 0);
  CHECK_LE(end_, value_capacity_);
  return *this;
}

size_t Slice::Size() const {
  CHECK(IsBounded());
  if (stride_ > 0 && start_ >= end_) { return 0; }
  if (stride_ < 0 && start_ <= end_) { return 0; }
  return ((end_ - start_) + (stride_ - ((stride_ > 0) - (stride_ < 0)))) / stride_;
}

bool Slice::IsContiguous() const {
  CHECK(IsBounded());
  return stride_ == 1;
}
bool Slice::IsCoveringAll() const {
  CHECK(IsBounded());
  return start_ == 0 && end_ == value_capacity_;
}

}  // namespace oneflow
